from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse


def indexing(texts):
    """ Imports CountVectorizer, TF-IDF Vectorizer and counts BM25  """
    k = 2
    b = 0.75
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    vectorizer = CountVectorizer()
    X_count = vectorizer.fit_transform(texts)
    X_tf = tf_vectorizer.fit_transform(texts)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    X_idf = tfidf_vectorizer.idf_
    l = X_count.sum(axis=1)
    avgdl = l.mean()
    data, row, col = [], [], []
    for i, j in zip(*X_tf.nonzero()):
        A = X_idf[j] * X_tf[i, j] * (k + 1)
        B = X_tf[i, j] + (k * (1 - b + b * l[i, 0] / avgdl))
        data.append(A / B)
        row.append(i)
        col.append(j)

    matrix = sparse.csr_matrix((data, (row, col)), shape=X_tf.shape)

    return matrix, vectorizer

