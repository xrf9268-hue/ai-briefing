import json, tempfile, os, numpy as np
from ai_briefing_ext.dedup.fingerprint import canonicalize, simhash, group_families, collapse_families
from ai_briefing_ext.dedup.semantic import ann_groups, merge_semantic

def test_canonicalize():
    assert canonicalize(" A  B ") == "A B"

def test_simhash_similarity():
    a = simhash("OpenAI builds models")
    b = simhash("OpenAI builds language models")
    assert a != 0 and b != 0

def test_family_collapse():
    items = [
        {"id":"1","text":"Hello world!","url":"u1","source":"s","ts":1},
        {"id":"2","text":"Hello  world!","url":"u2","source":"s","ts":2},
        {"id":"3","text":"Different text","url":"u3","source":"s","ts":3},
    ]
    fams = group_families(items, bits=64, bands=8, ham_thresh=3)
    out = collapse_families(items, fams)
    assert len(out) >= 2

def test_semantic_groups():
    items = [{"id":str(i),"text":f"doc {i}","url":f"u{i}","source":"s"} for i in range(5)]
    X = np.eye(5)  # orthogonal, no groups beyond self
    groups = ann_groups(X, threshold=0.99, topk=3)
    out = merge_semantic(items, groups)
    assert len(out) == 5
