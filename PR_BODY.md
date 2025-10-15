# feat(context-packer): token budgeting with per-cluster quotas

**Why**
- Keep LLM input within budget while maximizing information density & diversity.

**What**
- `ai_briefing_ext/packer/packer.py`: heuristic token estimator (uses `tiktoken` if present), per-cluster min/max quotas, Top-N by rerank scores, sentence-level de-dup and multi-source citation merge.
- `docs/DESIGN_PACKER.md`: policy & knobs.
- `tests/test_packer.py`.

**Back-compat**: Additive.

**Usage**
```bash
python -m ai_briefing_ext.packer.packer --in clusters_ranked.jsonl --out packed.json --budget 6000 --min 300 --max 1200
```
