import os
import sqlite3
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import update_newsboat_records


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
