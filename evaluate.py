import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from utils import to_poincare_ball, hyperbolic_distance, poincare_mean, rrf_fuse


def _average_precision(scores, labels):
    order = np.argsort(-np.array(scores))
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0:
            hits += 1
            total += hits / rank
    return total / max(hits, 1)


def evaluate_from_data(X_tfidf, X_smart, y, n_splits=5):
    X_tfidf = np.array(X_tfidf.todense() if hasattr(X_tfidf, "todense") else X_tfidf, dtype=np.float32)
    X_smart = np.array(X_smart, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    baseline_scores = []
    fused_scores = []
    for train_idx, test_idx in kf.split(X_tfidf, y):
        clf_tf = LinearSVC(class_weight="balanced", max_iter=1000000, tol=1e-6)
        clf_tf.fit(X_tfidf[train_idx], y[train_idx])
        clf_sm = LinearSVC(class_weight="balanced", max_iter=1000000, tol=1e-6)
        clf_sm.fit(X_smart[train_idx], y[train_idx])
        positives = [v for v, l in zip(X_smart[train_idx], y[train_idx]) if l > 0]
        centroid = poincare_mean(positives)
        s_tf = clf_tf.decision_function(X_tfidf[test_idx])
        s_sm = clf_sm.decision_function(X_smart[test_idx])
        base = s_sm * 0.65 + s_tf * 0.35
        baseline_scores.append(_average_precision(base, y[test_idx]))
        dists = [-hyperbolic_distance(to_poincare_ball(v), centroid) for v in X_smart[test_idx]]
        fused = rrf_fuse([s_tf.tolist(), s_sm.tolist(), dists])
        fused_scores.append(_average_precision(fused, y[test_idx]))
    return float(np.mean(baseline_scores)), float(np.mean(fused_scores))


def evaluate(db_path, n_splits=5):
    prefix = os.path.basename(db_path).split(".")[0]
    tfidf_path = os.path.join("models", prefix + "tfidf.p")
    data = pickle.load(open(tfidf_path, "rb"))
    X_tfidf = data["X_tfidf"]
    X_smart = data["X_smart"]
    y = data["y"]
    base, fused = evaluate_from_data(X_tfidf, X_smart, y, n_splits)
    print("Baseline MAP: %.4f" % base)
    print("Fused MAP: %.4f" % fused)


if __name__ == "__main__":
    evaluate(sys.argv[1])

