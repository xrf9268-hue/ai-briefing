# feat(dedup): two-stage dedup (fingerprint → semantic) with source merging

**Why**
- Reduce redundancy early with low-cost text-level fingerprints.
- Group near-duplicates semantically via ANN; keep *representatives* but **merge all sources** for citations.

**What**
- `ai_briefing_ext/dedup/fingerprint.py`: canonicalize text, SimHash + banded LSH families.
- `ai_briefing_ext/dedup/semantic.py`: ANN (FAISS if available; fallback to NumPy/Sklearn) with τ threshold.
- `ai_briefing_ext/dedup/io.py`: JSONL IO helpers and CLI.
- `docs/DESIGN_DEDUP.md`: design & thresholds.
- `tests/test_dedup.py`: synthetic tests.

**Back-compat**
- Additive only. No existing paths modified.

**Usage (CLI)**
```bash
# Fingerprint stage
python -m ai_briefing_ext.dedup.fingerprint --in items.jsonl --out items.fp.jsonl

# Semantic stage (requires vectors or an embedder endpoint)
python -m ai_briefing_ext.dedup.semantic --in items.fp.jsonl --out items.sem.jsonl --threshold 0.90
```
