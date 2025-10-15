import json, tempfile, os
from ai_briefing_ext.packer.packer import pack

def test_pack_budget():
    clusters = [{
        "topic_id": "c0",
        "label": "Test",
        "items": [{"text": "Sentence one. Sentence two. Sentence three.", "urls": ["u1"]}]
    }]
    out = pack(clusters, token_budget=100, per_cluster_min=10, per_cluster_max=50, title="T", date_iso="2025-01-01T00:00:00Z")
    assert "topics" in out and len(out["topics"])==1
