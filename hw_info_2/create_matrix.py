from sklearn.feature_extraction.text import TfidfVectorizer


def indexing(texts):
    """ Imports CountVectorizer to get matrix and other features for futher work  """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    vect = vectorizer.get_feature_names()
    return X, vect, vectorizer


