"""
Rerank with CrossEncoder (if available) + MMR diversity.
CLI:
  python -m ai_briefing_ext.rerank.rerank --in cluster_items.jsonl --out cluster_ranked.jsonl --lambda 0.7 --topn 5
Input format: JSONL per item with keys: id, text, (optional) vector, (optional) center_text or center_vector
"""
from __future__ import annotations
import argparse, json, typing as t
import numpy as np

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    HAS_XE = True
except Exception:
    HAS_XE = False
    CrossEncoder = None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

def mmr_rank(items: t.List[dict], lam: float = 0.7, topn: int = 5, model_name: str = "BAAI/bge-reranker-v2-m3") -> t.List[dict]:
    # Build relevance scores
    if HAS_XE:
        xe = CrossEncoder(model_name)
        pairs = [(it.get("center_text",""), it.get("text","")) for it in items]
        rel = np.array(xe.predict(pairs), dtype=float)
    else:
        # cosine to center vector
        center = None
        for it in items:
            v = it.get("center_vector")
            if v is not None:
                center = np.array(v, dtype=float); break
        if center is None:
            center = np.mean(np.array([it["vector"] for it in items], dtype=float), axis=0)
        rel = []
        for it in items:
            v = np.array(it["vector"], dtype=float)
            rel.append(cosine(center, v))
        rel = np.array(rel, dtype=float)
    # Precompute pairwise cosine for diversity
    vecs = np.array([it["vector"] for it in items], dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)+1e-9
    normed = vecs / norms
    sim_mat = normed @ normed.T
    selected = []
    cand = list(range(len(items)))
    # Initialize with best relevance
    i0 = int(np.argmax(rel))
    selected.append(i0); cand.remove(i0)
    while len(selected) < min(topn, len(items)) and cand:
        best_j, best_score = None, -1e9
        for j in cand:
            div_penalty = np.max(sim_mat[j, selected])
            score = lam*rel[j] - (1-lam)*div_penalty
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j); cand.remove(best_j)
    return [items[i] for i in selected]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8")]
    ranked = mmr_rank(items, lam=args.lam, topn=args.topn, model_name=args.model)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in ranked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"selected: {len(ranked)}")

if __name__ == "__main__":
    main()
