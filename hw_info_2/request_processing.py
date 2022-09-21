from preprocessing import lemmatize
from sklearn.metrics.pairwise import cosine_similarity


def request_search(vectorizer, text):
    """ Gets request, use function lemmitize() and create a vector """
    text_clean = [lemmatize(text)]
    vect_request = vectorizer.transform(text_clean).toarray()
    return vect_request


def search_similar(v1, v2):
    """ Counts cosine similarity """
    v_sim = cosine_similarity(v1, v2)
    return v_sim


def corpus_similar(parsed_texts, index_matrix, vec_request):
    """ Counts cosine similarity for corpus and prints a result """
    dict_doc_req = {}
    name_of_files = list(parsed_texts.keys())
    for i in range(0, index_matrix.shape[0]):
        dict_doc_req[name_of_files[i]] = search_similar(vec_request, index_matrix[i])
    result = dict(sorted(dict_doc_req.items(), key=lambda item: item[1], reverse=True))
    print('Наиболее близкие документы (по убыванию)')
    for key, item in result.items():
        print(key)

