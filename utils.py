import os
import pickle
import tempfile
import numpy as np
from contextlib import contextmanager
from sqlite3 import dbapi2 as sqlite3

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















