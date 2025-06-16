import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import evaluate_from_data


def test_evaluate_from_data_improves():
    X_tfidf = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    X_smart = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.85, 0.15],
        [0.15, 0.85],
    ])
    y = np.array([1, 0, 1, 0])
    base, fused = evaluate_from_data(X_tfidf, X_smart, y, n_splits=2)
    assert fused >= base
