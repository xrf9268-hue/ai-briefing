"""
Clustering with HDBSCAN (if available) or KMeans fallback. Noise attach strategy.
CLI:
  python -m ai_briefing_ext.clustering.cluster --in items.vec.jsonl --out clusters.jsonl --algo hdbscan --min_cluster_size 3 --attach 0.86
"""
from __future__ import annotations
import argparse, json, math, typing as t
import numpy as np

try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SK = True
except Exception:
    HAS_SK = False
    KMeans = None
    silhouette_score = None
    TfidfVectorizer = None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a @ b) / (na * nb))

def label_keywords(texts: t.List[str], topk: int = 4) -> str:
    if not HAS_SK:
        return ""
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words=None)
    X = vec.fit_transform(texts)
    idxs = np.array(X.sum(axis=0)).ravel().argsort()[::-1][:topk]
    feats = np.array(vec.get_feature_names_out())
    return " / ".join(feats[idxs])[:48]

def cluster_vectors(vectors: np.ndarray, algo: str = "hdbscan", min_cluster_size: int = 3) -> np.ndarray:
    n, d = vectors.shape
    if n == 0:
        return np.zeros((0,), dtype=int)
    if algo == "hdbscan" and HAS_HDBSCAN:
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = model.fit_predict(vectors)
        return labels
    # Fallback: KMeans with K from 2..min(8,n)
    if not HAS_SK:
        # trivial: one cluster
        return np.zeros((n,), dtype=int)
    best_labels = None
    best_score = -1.0
    for k in range(2, min(8, n)+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        l = km.fit_predict(vectors)
        s = silhouette_score(vectors, l) if len(set(l))>1 else -1.0
        if s > best_score:
            best_score, best_labels = s, l
    return best_labels if best_labels is not None else np.zeros((n,), dtype=int)

def attach_noise(vectors: np.ndarray, labels: np.ndarray, attach: float = 0.86) -> np.ndarray:
    # attach label -1 to nearest cluster center if cosine≥attach; else keep as singleton with new label
    out = labels.copy()
    valid = [c for c in set(labels.tolist()) if c != -1]
    if not valid:
        return out
    centers = {}
    for c in valid:
        centers[c] = vectors[labels==c].mean(axis=0)
    next_label = (max(valid)+1) if valid else 0
    for i, lab in enumerate(labels):
        if lab != -1:
            continue
        best, best_c = -1.0, None
        for c, cen in centers.items():
            s = cosine(vectors[i], cen)
            if s > best:
                best, best_c = s, c
        if best >= attach:
            out[i] = best_c
        else:
            out[i] = next_label
            centers[next_label] = vectors[i]
            next_label += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--algo", choices=["hdbscan","kmeans"], default="hdbscan")
    ap.add_argument("--min_cluster_size", type=int, default=3)
    ap.add_argument("--attach", type=float, default=0.86)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8")]
    X = np.array([it["vector"] for it in items], dtype=float)
    labels = cluster_vectors(X, algo=args.algo, min_cluster_size=args.min_cluster_size)
    labels = attach_noise(X, labels, attach=args.attach)
    # group and label
    out_clusters = {}
    for idx, lab in enumerate(labels.tolist()):
        out_clusters.setdefault(lab, []).append(items[idx])
    results = []
    for lab, members in out_clusters.items():
        texts = [(m.get("title") or "") + " " + (m.get("text") or "") for m in members]
        lbl = label_keywords(texts) if len(members)>=2 else (members[0].get("title") or "")
        results.append({"topic_id": f"c{lab}", "label": lbl, "members": [m.get("id") for m in members]})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"clusters: {len(results)}")

if __name__ == "__main__":
    main()
