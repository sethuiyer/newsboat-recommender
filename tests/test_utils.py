import os
import sqlite3
import tempfile
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    update_newsboat_records,
    rrf_fuse,
    hyperbolic_distance,
)


def create_temp_db():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE rss_item (id INTEGER PRIMARY KEY, flags TEXT)")
    conn.executemany(
        "INSERT INTO rss_item (id, flags) VALUES (?, ?)",
        [
            (1, None),
            (2, 's'),
            (3, None),
        ],
    )
    conn.commit()
    return conn, path


def test_update_newsboat_records():
    conn, path = create_temp_db()
    conn.close()
    update_newsboat_records(None, path, [1, 3])
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT id, flags FROM rss_item ORDER BY id").fetchall()
    assert rows[0][1] == 'rec'
    assert rows[1][1] == 's'
    assert rows[2][1] == 'rec'
    conn.close()
    os.remove(path)


def test_rrf_fuse_order():
    s1 = [0.9, 0.1, 0.8]
    s2 = [0.8, 0.2, 0.7]
    s3 = [0.3, 0.1, 0.2]
    fused = rrf_fuse([s1, s2, s3])
    assert fused[0] == max(fused)


def test_hyperbolic_distance_zero():
    u = [0.1, 0.2]
    v = [0.1, 0.2]
    assert hyperbolic_distance(u, v) == pytest.approx(0.0, abs=1e-6)


def _average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            total += hits / rank
    return total / sum(labels)


def test_fusion_improves():
    labels = [1, 0, 1, 0, 1]
    tfidf = [0.9, 0.8, 0.1, 0.3, 0.2]
    smart = [0.1, 0.2, 0.9, 0.4, 0.3]
    hyper = [0.4, 0.3, 0.5, 0.2, 0.1]
    baseline = [smart[i] * 0.65 + tfidf[i] * 0.35 for i in range(len(labels))]
    ap_baseline = _average_precision(baseline, labels)
    fused = rrf_fuse([tfidf, smart, hyper])
    ap_fused = _average_precision(fused, labels)
    assert ap_fused > ap_baseline

