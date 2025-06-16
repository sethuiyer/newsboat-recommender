import os
import sys
import pickle
from utils import connect_db, query_db, safe_pickle_dump

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sentence_transformers import SentenceTransformer
from torch.nn import Embedding, Linear
from torch.quantization import quantize_dynamic


db_path = sys.argv[1]
prefix = os.path.basename(db_path).split('.')[0]
tfidf_path = os.path.join('models', prefix + 'tfidf.p')
meta_path = os.path.join('models', prefix + 'tfidf_meta.p')
model_path = os.path.join('models', prefix + 'model.p')

max_train = 5000
max_features = 5000
max_recommendations = 50
cool_nlp_model = quantize_dynamic(SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device='cpu'), {Linear, Embedding})


def generate_tfidf_pickles():
    """Gets all the read articles and considers those articles flagged as 's' as 1 and rest as 0 
    and produces the embeddings
    """
    sqldb = connect_db(db_path)
    records = query_db(sqldb, '''select feedurl, author, id, title, content, flags from rss_item where unread=0 order by pubDate DESC;''')
    content_list = []
    outcome_list = []
    id_list = []
    title_list = []
    for record in records:
        # We should not judge the book by its cover
        content_list.append('||'+ record['feedurl'] + '|| \n ||' + record['author'] + '|| \n ||' + record['title'] + '|| \n' + record['content'])
        outcome_list.append((record['flags'] is not None and 'r' not in record['flags'] and 's' in record['flags']) * 1)
        id_list.append(record['id'])
        # Yes, we are judging the book by its cover but we are using the cool NLP model to judge
        title_list.append(record['title']) 
    print("Total %d feed items found" %(len(content_list)))
    print(content_list[0])
    # compute tfidf vectors with scikits
    v = TfidfVectorizer(input='content', 
            encoding='utf-8', decode_error='replace', strip_accents='unicode', 
            lowercase=True, analyzer='word', stop_words='english', 
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
            ngram_range=(1, 2), max_features = max_features, 
            norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
            max_df=1.0, min_df=1)
    v.fit(content_list)
    print("Projecting them to a mathematical space..")
    X_tfidf = v.transform(content_list)
    X_smart = cool_nlp_model.encode(title_list)
    out = {}
    out['X_tfidf'] = X_tfidf
    out['X_smart'] = X_smart
    out['y'] = outcome_list
    out['v'] = v
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
    """
        Given the embeddings, generate our preferences
        model using support vector machines
    """
    meta = pickle.load(open(meta_path, 'rb'))
    out = pickle.load(open(tfidf_path, 'rb'))
    X_tfidf = out['X_tfidf']
    X_tfidf = X_tfidf.todense().astype(np.float32)
    y = out['y']
    y = np.array(y).astype(np.float32)
    X_smart = out['X_smart']
    print('Learning your preferences...')
    clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=1000000, tol=1e-6, C=0.1)
    clf.fit(X_tfidf, y)
    beclf = LinearSVC(class_weight='balanced', verbose=False, max_iter=1000000, tol=1e-6)
    beclf.fit(X_smart, y)
    model = {}
    model['db_name'] = db_path
    model['clf'] = clf
    model['beclf'] = beclf
    safe_pickle_dump(model, model_path)


if __name__ == "__main__":
    generate_tfidf_pickles()
    build_model(meta_path, tfidf_path)
    
















