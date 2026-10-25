import json, pickle, zipfile, argparse
from pathlib import Path
import numpy as np
import embed as em


def l2n(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def otsu(x):
    x = np.sort(np.asarray(x, np.float64)); n = len(x); cs = np.cumsum(x)
    best, bt = -1.0, float(x.mean())
    for i in range(1, n):
        m0, m1 = cs[i - 1] / i, (cs[-1] - cs[i - 1]) / (n - i)
        v = (i / n) * ((n - i) / n) * (m0 - m1) ** 2
        if v > best:
            best, bt = v, 0.5 * (x[i - 1] + x[i])
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["SV"])
    a = ap.parse_args()
    root = Path(em.project_root())

    with zipfile.ZipFile(root / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for lang in (l.upper() for l in a.lang):
            paths = em.find_data(lang, None)
            emb_dir = root / "embeddings_final" / lang
            if not em.cache_ok(emb_dir, lang):
                em.embed_language(lang, paths["usages"], paths["definitions"], emb_dir)

            usages = em.read_jsonl(paths["usages"])
            with open(emb_dir / "usages.pkl", "rb") as f:
                emb = pickle.load(f)["emb"]
            with open(emb_dir / "definitions.pkl", "rb") as f:
                defs = pickle.load(f)["defs"]

            U = l2n(np.stack([emb[(u["word"], u["sentence_id"])] for u in usages]))
            G = {w: l2n(d["variants"]["colon"]) for w, d in defs.items()}
            score = np.array([U[i] @ G[u["word"]] for i, u in enumerate(usages)])
            preds = (score > otsu(score)).astype(int)
            print(f"[{lang}] {len(usages)} usages, predicted positive {preds.mean():.1%}")

            lines = "".join(json.dumps({"word": u["word"], "sentence_id": u["sentence_id"], "label": int(p)}) + "\n"
                            for u, p in zip(usages, preds))
            zf.writestr(f"{lang}_subtask2.jsonl", lines)
    print("Wrote", root / "submission.zip")


if __name__ == "__main__":
    main()
