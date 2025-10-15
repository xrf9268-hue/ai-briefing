import numpy as np
from ai_briefing_ext.rerank.rerank import mmr_rank

def test_mmr_rank_basic():
    items = []
    center = np.array([1.0,0.0])
    for i in range(5):
        v = np.array([1.0, 0.2*i])
        items.append({"id": str(i), "text": f"doc {i}", "vector": v.tolist(), "center_vector": center.tolist()})
    ranked = mmr_rank(items, lam=0.7, topn=3)
    assert len(ranked) == 3
