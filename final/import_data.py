import json
from transformers import AutoTokenizer, AutoModel
import pymorphy2
from nltk.corpus import stopwords
import pickle
import streamlit as st


@st.cache(show_spinner=False, allow_output_mutation=True)
def import_files():
    """ Gets the path to the .jsonl file. Returns a dictionary with question and answers from them """
    texts = json.load(open("data/texts_sample.txt"))
    with open('data/vectorizer.pickle', 'rb') as pkl:
        vectorizer = pickle.load(pkl)
    with open('data/bert_matrix.pickle', 'rb') as pkl:
        bert_matrix_answers = pickle.load(pkl)
    with open('data/matrix_bm25.pickle', 'rb') as pkl:
        matrix_bm25 = pickle.load(pkl)
    with open('data/matrix_tfidf.pickle', 'rb') as pkl:
        matrix_tfidf = pickle.load(pkl)
    with open('data/vectorizer_tfidf.pickle', 'rb') as pkl:
        vectorizer_tfidf = pickle.load(pkl)
    morph = pymorphy2.MorphAnalyzer()
    stoplist = stopwords.words('russian')
    return texts, vectorizer, bert_matrix_answers, matrix_bm25, morph, stoplist, matrix_tfidf, vectorizer_tfidf


def import_bert():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer, model
