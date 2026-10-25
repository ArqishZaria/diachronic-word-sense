import re, json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "pierluigic/xl-lexeme"
CACHE_VERSION = 4


def project_root() -> Path:
    for root in [Path("/kaggle/working"), Path("/content"), Path.cwd()]:
        if any((root / p).exists() for p in ("embeddings_final", "dev", "test", "usages.jsonl")):
            return root
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def find_data(lang, data_root=None):
    lang = lang.upper()
    roots = ([Path(data_root)] if data_root else []) + [project_root()]
    subdirs = ([f"{data_root}/{lang}"] if data_root else []) + [f"dev/{lang}", f"test/{lang}", lang, "."]
    found = {}
    for key, fname in (("usages", "usages.jsonl"), ("definitions", "definitions.jsonl"), ("labels", "subtask2.jsonl")):
        for base in roots:
            hit = next((base / sd / fname for sd in subdirs if (base / sd / fname).exists()), None)
            if hit:
                found[key] = str(hit)
                break
    return found


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")
    return SentenceTransformer(MODEL_NAME, device=device).eval()


def fix_span(text, start, end, word):
    span = text[start:end]
    stem = word[:3].lower()
    on_boundary = (start == 0 or not text[start - 1].isalnum()) and (end >= len(text) or not text[end].isalnum())
    if span and (stem in span.lower() or (not any(c.isspace() for c in span) and on_boundary)):
        return start, end
    cands = [m for m in re.finditer(r"\w+", text) if m.group(0).lower().startswith(stem)]
    if not cands:
        return start, end
    m = min(cands, key=lambda m: abs(m.start() - start))
    return m.start(), m.end()


def prepare(tok, text, start, end, max_len):
    left = tok.tokenize(text[:start])
    span = tok.tokenize("<t>") + tok.tokenize(text[start:end]) + tok.tokenize("</t>")
    right = tok.tokenize(text[end:])
    budget = max_len - 2
    if len(left) + len(span) + len(right) > budget:
        room = max(budget - len(span), 0)
        l_allow, r_allow = room // 2, room - room // 2
        if len(left) < l_allow:
            r_allow += l_allow - len(left)
            l_allow = len(left)
        if len(right) < r_allow:
            l_allow += r_allow - len(right)
            r_allow = len(right)
        left = left[len(left) - l_allow:]
        right = right[:r_allow]
    return tok.convert_tokens_to_ids([tok.cls_token] + left + span + right + [tok.sep_token])


def encode(model, items, bs=32):
    tok = model.tokenizer
    prepared = [prepare(tok, t, s, e, model.max_seq_length) for t, s, e in items]
    out = [None] * len(items)
    order = np.argsort([-len(p) for p in prepared])
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(prepared[i]) for i in idx)
        ids = [prepared[i] + [tok.pad_token_id] * (L - len(prepared[i])) for i in idx]
        att = [[1] * len(prepared[i]) + [0] * (L - len(prepared[i])) for i in idx]
        feats = {"input_ids": torch.tensor(ids, device=model.device),
                 "attention_mask": torch.tensor(att, device=model.device)}
        with torch.no_grad():
            se = model(feats)["sentence_embedding"].cpu().numpy()
        for k, i in enumerate(idx):
            out[i] = se[k]
    return np.stack(out)


def _meta(lang):
    return {"version": CACHE_VERSION, "model": MODEL_NAME, "lang": lang}


def cache_ok(out_dir, lang):
    try:
        for name in ("usages.pkl", "definitions.pkl"):
            with open(Path(out_dir) / name, "rb") as f:
                if pickle.load(f).get("meta") != _meta(lang):
                    return False
        return True
    except Exception:
        return False


def embed_language(lang, usages_path, definitions_path, out_dir, model=None, force=False):
    lang, out_dir = lang.upper(), Path(out_dir)
    if cache_ok(out_dir, lang) and not force:
        print(f"[{lang}] embeddings found in {out_dir} (use --force to recompute)")
        return str(out_dir)
    model = model or load_model()

    usages = read_jsonl(usages_path)
    items, n_fixed = [], 0
    for u in usages:
        s, e = fix_span(u["text"], u["start"], u["end"], u["word"])
        n_fixed += (s, e) != (u["start"], u["end"])
        items.append((u["text"], s, e))
    print(f"[{lang}] {len(usages)} usages ({n_fixed} misaligned spans repaired)")
    vecs = encode(model, items)
    emb = {(u["word"], u["sentence_id"]): vecs[i] for i, u in enumerate(usages)}
    info = {(u["word"], u["sentence_id"]): {"period_label": u.get("period_label"), "year": u.get("year")}
            for u in usages}

    defs = {}
    if definitions_path:
        text = {}
        for d in read_jsonl(definitions_path):
            text.setdefault(d["word"], d["definition"].strip().rstrip(".").strip())
        words = list(text)
        if words:
            v = encode(model, [(f"{w}: {text[w]}", 0, len(w)) for w in words])
            defs = {w: {"definition": text[w], "variants": {"colon": v[i]}} for i, w in enumerate(words)}
        print(f"[{lang}] {len(defs)} definitions embedded")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "usages.pkl", "wb") as f:
        pickle.dump({"meta": _meta(lang), "emb": emb, "info": info}, f)
    with open(out_dir / "definitions.pkl", "wb") as f:
        pickle.dump({"meta": _meta(lang), "defs": defs}, f)
    return str(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["SV"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_root = Path(a.out_root) if a.out_root else project_root() / "embeddings_final"
    model = None
    for lang in (l.upper() for l in a.lang):
        paths = find_data(lang, a.data_root)
        if "usages" not in paths:
            print(f"[{lang}] usages.jsonl not found, skipping")
            continue
        if model is None and (a.force or not cache_ok(out_root / lang, lang)):
            model = load_model()
        embed_language(lang, paths["usages"], paths.get("definitions"), out_root / lang, model, a.force)
    print("Done.")


if __name__ == "__main__":
    main()
