# Dedup Design

Stages:
1) Fingerprint families: canonicalize → exact hash → SimHash (64-bit) → banded buckets (default: 8 bands × 8 bits).
2) Semantic neighbors: if vectors exist: use FAISS (preferred) or cosine similarity fallback.
   If vectors are missing: you can attach an embedder callable (TEI/OpenAI/Gemini) before this stage.

Representative selection:
- score = w_len * normalized_length + w_src * source_trust + w_ts * recency
- keep the best; merge all `urls/sources/ids` into `merged_sources` for later citations.

Thresholds:
- Fingerprint Hamming distance ≤ 3 → same family (tuneable)
- Semantic τ_sem in [0.88, 0.92] typical

Outputs:
- Items with `family_id`, `rep=True/False`, `merged_sources=[... ]` after collapse.
