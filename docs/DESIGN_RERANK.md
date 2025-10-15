# Rerank + MMR

- Query = cluster center (mean vector or representative text).
- Relevance: CrossEncoder score if available; else cosine to center.
- Diversity: penalize candidates similar to already selected.
- Output: top-N ids per cluster in ranked order.
