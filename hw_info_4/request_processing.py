from preprocessing import lemmatize
import pymorphy2
from nltk.corpus import stopwords
import numpy as np
import torch
from create_matrix_bert import mean_pooling
from sklearn.preprocessing import normalize


def request_search_bm25(vectorizer, text):
    """ Gets request, use function lemmitize() and create a vector """
    morph = pymorphy2.MorphAnalyzer()
    stoplist = stopwords.words('russian')
    text_clean = [lemmatize(text, morph, stoplist)]
    vect_request = vectorizer.transform(text_clean)
    return vect_request


def request_search_bert(tokenizer, model, text):
    """ Gets request, use function lemmitize() and create a vector """
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    vect_request = mean_pooling(output, encoded_input['attention_mask'])
    return normalize(vect_request)


def search_similar(matrix, request):
    """ Counts similarity matrix and request """
    v_sim = np.dot(matrix, request.T)
    return v_sim


def create_output(scores, texts):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    result = np.array(texts)[sorted_scores_indx.ravel()]
    return result




