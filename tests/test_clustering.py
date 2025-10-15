import json, numpy as np
from ai_briefing_ext.clustering.cluster import cluster_vectors, attach_noise

def test_attach_noise_singleton():
    X = np.array([[1,0],[0.9,0.1],[0,1]], dtype=float)
    labels = np.array([0,0,-1], dtype=int)
    out = attach_noise(X, labels, attach=0.5)
    assert out[2] != -1
