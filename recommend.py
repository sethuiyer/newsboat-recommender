import os
import pickle
import tempfile
import numpy as np
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sqlite3 import dbapi2 as sqlite3
import sys 

db_path = sys.argv[1]

tfidf_path = 'tfidf.p'
meta_path = 'tfidf_meta.p'
model_path = 'model.p'
max_train = 5000
max_features = 5000

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

def generate_tfidf_pickles():
    sqldb = connect_db(db_path)
    records = query_db(sqldb, 'select feedurl, author, id, title, content, flags from rss_item')
    content_list = []
    outcome_list = []
    id_list = []
    for record in records:
        content_list.append('||'+ record['feedurl'] + '|| \n ||' + record['author'] + '|| \n ||' + record['title'] + '|| \n' + record['content'])
        outcome_list.append((record['flags'] is not None and 'r' not in record['flags'] and 's' in record['flags']) * 1)
        id_list.append(record['id'])
    print("Total %d feed items found" %(len(content_list)))
    print(content_list[0])
    # compute tfidf vectors with scikits
    v = TfidfVectorizer(input='content', 
            encoding='utf-8', decode_error='replace', strip_accents='unicode', 
            lowercase=True, analyzer='word', stop_words='english', 
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
            ngram_range=(1, 3), max_features = max_features, 
            norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
            max_df=1.0, min_df=1)
    v.fit(content_list)
    print("Projecting them to a mathematical space..")
    X = v.transform(content_list)
    #print(X.shape)
    out = {}
    out['X'] = X
    out['y'] = outcome_list
    #print("writing", tfidf_path)
    safe_pickle_dump(out, tfidf_path)
    out = {}
    out['vocab'] = v.vocabulary_
    out['idf'] = v._tfidf.idf_
    out['ids'] = id_list
    out['idtoi'] = {x:i for i,x in enumerate(id_list)}
    #print("Writing Meta Data")
    safe_pickle_dump(out, meta_path)

def build_model(meta_path, tfidf_path):
    meta = pickle.load(open(meta_path, 'rb'))
    out = pickle.load(open(tfidf_path, 'rb'))
    X = out['X']
    X = X.todense().astype(np.float32)
    y = out['y']
    y = np.array(y).astype(np.float32)
    print('Learning your preferences...')
    clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
    clf.fit(X, y)
    model = {}
    model['db_name'] = db_path
    model['clf'] = clf
    safe_pickle_dump(model, model_path)

def generate_recs_from_model(meta_path, tfidf_path, model_path):
    print("Generating Recommendations...")
    meta = pickle.load(open(meta_path, 'rb'))
    out = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    clf = model['clf']
    X = out['X']
    X = X.todense().astype(np.float32)
    y = out['y']
    y = np.array(y).astype(np.float32)
    s = clf.decision_function(X)
    sortix = np.argsort(-s)
    recs = sortix[y[sortix] == 0]
    recs = recs[:30]
    #print(recs)
    return recs

def update_newsboat_records(meta_path, db_path, recs):
    meta = pickle.load(open(meta_path, 'rb'))
    print("Updating the database...")
    sqldb = sqlite3.connect(db_path)
    for rec in recs:
        sqldb.execute(
            """UPDATE {} SET unread=?, flags=? WHERE id = ?""".format('rss_item'),
            (1, 'rec', meta['ids'][rec])
        )
        sqldb.commit()
        

if __name__ == "__main__":
    generate_tfidf_pickles()
    build_model(meta_path, tfidf_path)
    recs = generate_recs_from_model(meta_path, tfidf_path, model_path)
    update_newsboat_records(meta_path, db_path, recs)
    
















