import os
import sys
import pickle
from utils import (
    connect_db,
    query_db,
    update_newsboat_records,
    to_poincare_ball,
    hyperbolic_distance,
    rrf_fuse,
)

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
max_recommendations = 30
cool_nlp_model = quantize_dynamic(SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device='cpu'), {Linear, Embedding})


def generate_recs_from_model(meta_path, tfidf_path, model_path):
    print("Generating Recommendations...")
    sqldb = connect_db(db_path)
    records = query_db(sqldb, '''select feedurl, author, id, title, content, flags from rss_item order by pubDate DESC LIMIT 200;''')
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
    #Loading the pickle files
    meta = pickle.load(open(meta_path, 'rb'))
    out = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    v = out['v']
    print("Projecting them to a mathematical space..")
    X_tfidf = v.transform(content_list)
    X_smart = cool_nlp_model.encode(title_list)
    clf = model['clf']
    beclf = model['beclf']
    centroid = model.get('centroid')
    y = np.array(outcome_list).astype(np.float32)
    X_tfidf = X_tfidf.todense().astype(np.float32)
    print("Recommending...")
    s_tfidf = clf.decision_function(X_tfidf)
    s_smart = beclf.decision_function(X_smart)
    if centroid is not None:
        poincare_vectors = [to_poincare_ball(v) for v in X_smart]
        dists = [ -hyperbolic_distance(v, centroid) for v in poincare_vectors]
        fusion = rrf_fuse([s_tfidf.tolist(), s_smart.tolist(), dists])
    else:
        fusion = rrf_fuse([s_tfidf.tolist(), s_smart.tolist()])
    sortix = np.argsort(-np.array(fusion))
    recs = sortix[y[sortix] == 0]
    recs = recs[:max_recommendations]
    print(recs)
    print([id_list[x] for x in recs])
    return [id_list[x] for x in recs]


if __name__ == "__main__":
    recs = generate_recs_from_model(meta_path, tfidf_path, model_path)
    update_newsboat_records(meta_path, db_path, recs)
    
















