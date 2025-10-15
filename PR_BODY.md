# feat(clustering): hdbscan/kmeans with noise attach + lightweight topic labels

**Why**
- Improve topic purity and avoid mixing noise (-1) with valid clusters.

**What**
- `ai_briefing_ext/clustering/cluster.py`: HDBSCAN (if installed) or KMeans fallback; **noise attach** to nearest centroid with τ_attach; optional TF-IDF based labels.
- `docs/DESIGN_CLUSTERING.md`: parameters & guidance.
- `tests/test_clustering.py`: synthetic tests.

**Back-compat**
- Additive; no existing paths changed.

**Usage**
```bash
python -m ai_briefing_ext.clustering.cluster --in items.vec.jsonl --out clusters.jsonl --algo hdbscan --min_cluster_size 3 --attach 0.86
```
