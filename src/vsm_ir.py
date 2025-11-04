import math
from collections import Counter

def compute_tf_df(docs):
    tf = {}
    df = Counter()
    for d, toks in docs.items():
        tf[d] = Counter(toks)
        for t in set(toks):
            df[t] += 1
    return tf, df

def compute_idf(df, N):
    return {t: math.log((N)/(df_t + 1e-12)) for t, df_t in df.items()}

def tfidf_vector(tf_counter, idf, sublinear=False):
    vec = {}
    for t, f in tf_counter.items():
        if sublinear:
            w_tf = 1 + math.log(f) if f > 0 else 0.0
        else:
            w_tf = float(f)
        vec[t] = w_tf * idf.get(t, 0.0)
    return vec

def cosine_sim(vec_q, vec_d):
    dot = 0.0
    for t, w in vec_q.items():
        if t in vec_d:
            dot += w * vec_d[t]
    nq = math.sqrt(sum(w*w for w in vec_q.values())) + 1e-12
    nd = math.sqrt(sum(w*w for w in vec_d.values())) + 1e-12
    return dot / (nq * nd)

def build_tfidf_index(docs, sublinear=False):
    tf, df = compute_tf_df(docs)
    N = len(docs)
    idf = compute_idf(df, N)
    doc_vecs = {d: tfidf_vector(tf[d], idf, sublinear=sublinear) for d in docs}
    return doc_vecs, idf

def vectorize_query(query_tokens, idf, sublinear=False):
    tfq = Counter(query_tokens)
    return tfidf_vector(tfq, idf, sublinear=sublinear)

def rank_docs(query_tokens, doc_vecs, idf, k=5, sublinear=False):
    qv = vectorize_query(query_tokens, idf, sublinear=sublinear)
    scores = []
    for d, dv in doc_vecs.items():
        s = cosine_sim(qv, dv)
        scores.append((d, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
