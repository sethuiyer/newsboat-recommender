import os
import pickle
import tempfile
from contextlib import contextmanager
from sqlite3 import dbapi2 as sqlite3

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy may not be installed in tests
    _HAS_NUMPY = False

#Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def _tempfile(*args, **kws):
    """ Context for temporary file.
    Will find a free temporary filename upon entering
    and will try to delete the file on leaving
    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.
    Allows reading and writing to and from the same filename.
    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    # Ensure the target directory exists before creating the temporary file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)

def safe_pickle_dump(obj, fname):
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1)


def connect_db(db_path):
  sqlite_db = sqlite3.connect(db_path)
  sqlite_db.row_factory = sqlite3.Row # to return dicts rather than tuples
  return sqlite_db


def query_db(db, query, args=()):
  """Queries the database and returns a list of dictionaries."""
  cur = db.execute(query, args)
  rv = cur.fetchall()
  return rv

def update_newsboat_records(meta_path, db_path, recs):
    print("Updating the database...")
    sqldb = sqlite3.connect(db_path)
    for rec in recs:
        sqldb.execute(
            """UPDATE {} SET flags=? WHERE id = ?""".format('rss_item'),
            ('rec', rec)
        )
        sqldb.commit()


def _norm(vec):
    if _HAS_NUMPY:
        return float(np.linalg.norm(vec))
    return float(sum(x * x for x in vec) ** 0.5)


def to_poincare_ball(vec):
    norm = _norm(vec)
    denom = 1 + (1 + norm * norm) ** 0.5
    if _HAS_NUMPY:
        return vec / denom
    return [x / denom for x in vec]


def hyperbolic_distance(u, v, eps=1e-9):
    if len(u) != len(v):
        raise ValueError("Vectors must have the same dimension")
    nu = _norm(u)
    nv = _norm(v)
    diff = [(a - b) for a, b in zip(u, v)]
    nd = _norm(diff) ** 2
    denom = (1 - nu ** 2) * (1 - nv ** 2)
    if denom <= 0:
        denom = eps
    val = 1 + 2 * nd / denom
    if val < 1:
        val = 1.0
    import math
    return math.acosh(val)


def poincare_mean(vectors):
    if not vectors:
        raise ValueError("Empty vector list")
    if _HAS_NUMPY:
        arr = np.array(vectors)
        mean = arr.mean(axis=0)
    else:
        dim = len(vectors[0])
        mean = [0.0] * dim
        for vec in vectors:
            for i, x in enumerate(vec):
                mean[i] += x
        mean = [x / len(vectors) for x in mean]
    return to_poincare_ball(mean)


def rrf_fuse(score_lists, k=60):
    if not score_lists:
        return []
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0] * n
        for idx, doc in enumerate(order, start=1):
            ranks[doc] = idx
        for i in range(n):
            fused[i] += 1.0 / (k + ranks[i])
    return fused















