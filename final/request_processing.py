from string import punctuation, digits
import numpy as np
import torch
from sklearn.preprocessing import normalize

KNOWN_WORDS = {}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def lemmatize(text, morph, stoplist):
    """ Cleans texts, gets lemma from pymorphy2 and return text with lemmas"""
    text = [word.lower().strip(punctuation) for word in text.split()]
    text = [word.translate(str.maketrans('', '', digits)) for word in text]
    text = [word for word in text if word != '']
    text = [word for word in text if word not in stoplist]
    lemmas = []
    global KNOWN_WORDS  # храним все разборы для повышения скорости работы, pymorphy2 не учитывает контекст
    for word in text:
        if word in KNOWN_WORDS:
            lemmas.append(KNOWN_WORDS[word])
        else:
            result = morph.parse(word)[0].normal_form
            lemmas.append(result)
            KNOWN_WORDS[word] = result
    return ' '.join(lemmas)


def request_search_tfidf(vectorizer, text, morph, stoplist):
    """ Gets request, use function lemmitize() and create a vector """
    text_clean = [lemmatize(text, morph, stoplist)]
    vect_request = vectorizer.transform(text_clean)
    return vect_request


def request_search_bm25(vectorizer, text, morph, stoplist):
    """ Gets request, use function lemmitize() and create a vector """
    text_clean = [lemmatize(text, morph, stoplist)]
    vect_request = vectorizer.transform(text_clean)
    return vect_request


def request_search_bert(tokenizer, model, text):
    """ Gets request, use function lemmitize() and create a vector """
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
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




