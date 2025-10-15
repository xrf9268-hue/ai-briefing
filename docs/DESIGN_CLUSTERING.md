# Clustering Design

- Default: HDBSCAN(metric='euclidean' on vectors); Fallback: KMeans with automatic K via silhouette.
- Noise handling: do NOT mix -1s into an 'other' heap; try attach to nearest cluster if similarity ≥ τ_attach; else keep as singleton.
- Labels: top 2~4 keywords from titles/texts via TF-IDF, length 3~12 chars, no hype words.
