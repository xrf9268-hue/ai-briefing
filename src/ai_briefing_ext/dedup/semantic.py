"""
Semantic dedup via ANN (FAISS if available; otherwise NumPy cosine).
CLI:
  python -m ai_briefing_ext.dedup.semantic --in items.fp.jsonl --out items.sem.jsonl --threshold 0.90
Assumes items either carry 'vector' (list[float]) or you inject vectors before calling.
"""
from __future__ import annotations
import argparse, json, math
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def ann_groups(vectors: np.ndarray, threshold: float = 0.90, topk: int = 20) -> List[List[int]]:
    n, d = vectors.shape
    if n == 0:
        return []
    if HAS_FAISS:
        index = faiss.IndexFlatIP(d)
        # normalize for cosine via inner product
        x = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        index.add(x.astype(np.float32))
        D, I = index.search(x.astype(np.float32), min(topk, n))
        groups = []
        visited = set()
        for i in range(n):
            if i in visited: 
                continue
            neigh = [j for j, s in zip(I[i].tolist(), D[i].tolist()) if j != i and s >= threshold]
            group = [i] + neigh
            for j in neigh:
                visited.add(j)
            groups.append(sorted(set(group)))
        return groups
    else:
        # cosine via numpy
        sims = _cosine_sim(vectors, vectors)
        groups = []
        visited = set()
        for i in range(n):
            if i in visited: 
                continue
            neigh = [j for j in range(n) if j != i and sims[i, j] >= threshold]
            group = [i] + neigh
            for j in neigh:
                visited.add(j)
            groups.append(sorted(set(group)))
        return groups

def merge_semantic(items: List[Dict], groups: List[List[int]]) -> List[Dict]:
    used = set()
    out = []
    for gid, g in enumerate(groups):
        rep_idx = max(g, key=lambda k: len((items[k].get("text") or "")))
        rep = dict(items[rep_idx])
        rep["semantic_group_id"] = f"sg_{gid}"
        rep["rep"] = True
        # merge all sources/urls (dedup by url)
        urls = set()
        merged = []
        for j in g:
            it = items[j]
            for src in it.get("merged_sources", []) or [{"id": it.get("id"), "url": it.get("url"), "source": it.get("source")}]:  # fallback
                key = src.get("url") or src.get("id")
                if key and key not in urls:
                    urls.add(key)
                    merged.append(src)
            used.add(j)
        rep["merged_sources"] = merged
        out.append(rep)
    # add stragglers
    for k, it in enumerate(items):
        if k in used: 
            continue
        it2 = dict(it)
        it2.setdefault("merged_sources", [{"id": it.get("id"), "url": it.get("url"), "source": it.get("source")}])
        out.append(it2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8")]
    # Extract vectors
    vecs = []
    for it in items:
        v = it.get("vector")
        if v is None:
            raise SystemExit("Vector field missing. Please attach embeddings before semantic stage.")
        vecs.append(v)
    X = np.array(vecs, dtype=float)
    groups = ann_groups(X, threshold=args.threshold, topk=args.topk)
    out = merge_semantic(items, groups)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"semantic groups: {len(groups)}, out: {len(out)}")

if __name__ == "__main__":
    main()
