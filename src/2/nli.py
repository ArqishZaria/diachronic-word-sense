import json, zipfile, argparse, hashlib, pickle
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import embed as em

MODEL = "MoritzLaurer/bge-m3-zeroshot-v2.0"
TEMPLATES = ['The word "{w}" is used here to mean: {d}.',
             'In this sentence, "{w}" has the meaning: {d}.']
MU = 0.3


def load(lang, emb_dir):
    paths = em.find_data(lang, None)
    usages = em.read_jsonl(paths["usages"])
    defs = {d["word"]: d["definition"].strip().rstrip(".").strip() for d in em.read_jsonl(paths["definitions"])}
    y = None
    if "labels" in paths:
        lab = {(r["word"], r["sentence_id"]): int(r["label"]) for r in em.read_jsonl(paths["labels"])}
        if all((u["word"], u["sentence_id"]) in lab for u in usages):
            y = np.array([lab[(u["word"], u["sentence_id"])] for u in usages])
    with open(emb_dir / "usages.pkl", "rb") as f:
        emb = pickle.load(f)["emb"]
    E = np.stack([emb[(u["word"], u["sentence_id"])] for u in usages])
    return usages, defs, y, E


def nli_scores(usages, defs, cache, ctx=250):
    if cache.exists():
        return np.load(cache)["L"]
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    ent = next((int(k) for k, v in model.config.id2label.items() if str(v).lower().startswith("entail")), 0)
    P, S, D = [], [], []
    for u in usages:
        t = u["text"]
        s, e = em.fix_span(t, u["start"], u["end"], u["word"])
        P.append((t[max(0, s - ctx):s] + '"' + t[s:e] + '"' + t[e:e + ctx]).strip())
        S.append(t[s:e])
        D.append(defs[u["word"]])
    order = np.argsort([-len(p) for p in P])
    L = np.zeros((len(TEMPLATES), len(P)))
    with torch.no_grad():
        for ti, T in enumerate(TEMPLATES):
            H = [T.format(w=a, d=d) for a, d in zip(S, D)]
            for b in range(0, len(order), 32):
                idx = order[b:b + 32]
                enc = tok([P[i] for i in idx], [H[i] for i in idx], truncation="only_first",
                          max_length=256, padding=True, return_tensors="pt").to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                    lg = model(**enc).logits.float()
                keep = [j for j in range(lg.shape[-1]) if j != ent]
                L[ti, idx] = (lg[:, ent] - torch.logsumexp(lg[:, keep], dim=-1)).cpu().numpy()
    np.savez(cache, L=L)
    return L


def otsu(x):
    x = np.sort(np.asarray(x, float)); n = len(x); cs = np.cumsum(x)
    best, bt = -1.0, float(x.mean())
    for i in range(1, n):
        m0, m1 = cs[i - 1] / i, (cs[-1] - cs[i - 1]) / (n - i)
        v = (i / n) * ((n - i) / n) * (m0 - m1) ** 2
        if v > best:
            best, bt = v, 0.5 * (x[i - 1] + x[i])
    return bt


def pca_block(A, npc, whiten):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    X = PCA(min(npc, len(A) - 1, A.shape[1]), random_state=0).fit_transform(A)
    if whiten:
        return (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X / (np.sqrt((X ** 2).sum(1).mean()) + 1e-9)


def smooth(E, s, groups):
    out = s.copy()
    for ix in groups.values():
        n = len(ix)
        if n < 30:
            continue
        X, sw = pca_block(E[ix], 20, False), s[ix]
        acc, cnt = np.zeros(n), 0
        for k in (2, 3, 4, 5, 6, 8, 10):
            if k * 5 > n:
                continue
            for sd in (0, 1, 2):
                lab = KMeans(k, n_init=1, random_state=sd).fit_predict(X)
                acc += (np.bincount(lab, weights=sw) / np.bincount(lab))[lab]
                cnt += 1
        out[ix] = acc / cnt
    return out


def self_train(u, E, groups, rounds=2, C=0.3):
    base = np.zeros(len(u))
    for ix in groups.values():
        base[ix] = u[ix] - otsu(u[ix])
    sd = base.std() + 1e-9
    cur, Xs = base, {}
    for _ in range(rounds):
        out = base / sd
        for w, ix in groups.items():
            if len(ix) < 40:
                continue
            m = cur[ix]
            pos, neg = m > 0, m <= 0
            if pos.sum() < 5 or neg.sum() < 5:
                continue
            pl = np.full(len(ix), -1)
            pl[pos & (m >= np.median(m[pos]))] = 1
            pl[neg & (m <= np.median(m[neg]))] = 0
            if w not in Xs:
                Xs[w] = pca_block(E[ix], 24, True)
            X, pred = Xs[w], np.zeros(len(ix))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(ix):
                t = tr[pl[tr] >= 0]
                if len(set(pl[t])) < 2:
                    continue
                pred[te] = LogisticRegression(C=C, max_iter=500).fit(X[t], pl[t]).decision_function(X[te])
            out[ix] = base[ix] / sd + pred / (pred.std() + 1e-9)
        cur = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["SV"])
    a = ap.parse_args()
    root = Path(em.project_root())

    with zipfile.ZipFile(root / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for lang in (l.upper() for l in a.lang):
            usages, defs, y, E = load(lang, root / "embeddings_final" / lang)
            words = np.array([u["word"] for u in usages])
            groups = {w: np.where(words == w)[0] for w in np.unique(words)}
            key = hashlib.md5(f"{MODEL}|{lang}|{len(usages)}|{TEMPLATES}".encode()).hexdigest()[:10]
            s = nli_scores(usages, defs, root / f"nli_unsup_{key}.npz").mean(0)
            sm = smooth(E, s, groups)
            preds = (self_train(sm + MU * (s - sm), E, groups) > 0).astype(int)
            print(f"[{lang}] positive {preds.mean():.1%}" + (f", F1 = {f1_score(y, preds):.4f}" if y is not None else ""))
            zf.writestr(f"{lang}_subtask2.jsonl", "".join(
                json.dumps({"word": u["word"], "sentence_id": u["sentence_id"], "label": int(p)}) + "\n"
                for u, p in zip(usages, preds)))
    print("Wrote", root / "submission.zip")


if __name__ == "__main__":
    main()