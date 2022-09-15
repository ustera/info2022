""" Main functions for processing text """

import pymorphy2
from string import punctuation, digits
from nltk.corpus import stopwords


def lemmatize(text):
    """ Cleans texts, gets lemma from pymorphy2 and return text with lemmas"""
    morph = pymorphy2.MorphAnalyzer()
    stoplist = stopwords.words('russian')
    text = text.replace('\ufeff', '')
    text = [word.lower().strip(punctuation) for word in text.split()]
    text = [word.translate(str.maketrans('', '', digits)) for word in text]
    text = [word for word in text if word != '']
    text = [word for word in text if word not in stoplist]
    lemmas = []
    global known_words  # храним все разборы для повышения скорости работы, pymorphy2 не учитывает контекст
    for word in text:
        if word in known_words:
            lemmas.append(known_words[word])
        else:
            result = morph.parse(word)[0].normal_form
            lemmas.append(result)
            known_words[word] = result
    return ' '.join(lemmas)


def create_dict(texts):
    """ Create dictionary for text (clean) storage  """
    parsed_texts = {}
    for key, value in texts.items():
        parsed_text = lemmatize(value)
        parsed_texts[key] = parsed_text
    return parsed_texts


known_words = {}
