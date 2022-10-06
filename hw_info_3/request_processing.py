from preprocessing import lemmatize
import pymorphy2
from nltk.corpus import stopwords
import numpy as np


def request_search(vectorizer, text):
    """ Gets request, use function lemmitize() and create a vector """
    morph = pymorphy2.MorphAnalyzer()
    stoplist = stopwords.words('russian')
    text_clean = [lemmatize(text, morph, stoplist)]
    vect_request = vectorizer.transform(text_clean)
    return vect_request


def search_similar(matrix, request):
    """ Counts similarity matrix and request """
    v_sim = np.dot(matrix, request.T)
    return v_sim.toarray()


def create_output(scores, texts):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    result = np.array(texts)[sorted_scores_indx.ravel()]
    return result


