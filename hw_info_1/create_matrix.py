from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def indexing(texts):
    """ Imports CountVectorizer to get matrix and other features for futher work  """
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(texts)
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    vect = vectorizer.get_feature_names()
    return X, matrix_freq, vect, vectorizer


