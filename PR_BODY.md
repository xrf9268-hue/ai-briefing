# feat(rerank): cross-encoder rerank + MMR inside clusters

**Why**
- Improve representativeness and diversity before packing to LLM.

**What**
- `ai_briefing_ext/rerank/rerank.py`: attempts `sentence_transformers` CrossEncoder; falls back to embedding cosine.
- Implements **MMR**: `score = λ*rel − (1−λ)*max_sim_to_selected`.
- `docs/DESIGN_RERANK.md` and `tests/test_rerank.py`.

**Back-compat**: Additive.

**Usage**
```bash
python -m ai_briefing_ext.rerank.rerank --in cluster_items.jsonl --out cluster_ranked.jsonl --lambda 0.7 --topn 5
```
