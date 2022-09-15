import numpy as np


def each_in_collection(index_matrix, dictionary):
    """ Looks for a set of words that occurs in all documents in the collection """
    res = np.sum(index_matrix.toarray(), 0)
    ind_res = []
    for i in range(len(res)):
        if res[i] == 165:
            ind_res.append(i)
    for ind in ind_res:
        return 'Набор слов, который есть во всех документах коллекции: ' + dictionary[ind]


def most_popular_person(matrix_freq, vectorizer):
    """ Create a list with names and search looks for occurrences of each name. Then finds
    in the dictionary the character with the largest number of occurrences """
    index_of_pers = {'Моника': vectorizer.vocabulary_.get('моника'),
                     'Мон': vectorizer.vocabulary_.get('мон'),
                     'Рэйчел': vectorizer.vocabulary_.get('рэйчел'),
                     'Рейч': vectorizer.vocabulary_.get('рейч'),
                     'Чендлер': vectorizer.vocabulary_.get('чендлер'),
                     'Чэндлер': vectorizer.vocabulary_.get('чэндлер'),
                     'Чен': vectorizer.vocabulary_.get('чен'),
                     'Фиби': vectorizer.vocabulary_.get('фиби'),
                     'Фибс': vectorizer.vocabulary_.get('фибс'),
                     'Росс': vectorizer.vocabulary_.get('росс'),
                     'Джоуи': vectorizer.vocabulary_.get('джоуи'),
                     'Джои': vectorizer.vocabulary_.get('джои'),
                     'Джо': vectorizer.vocabulary_.get('джо')}
    count_pers = {}
    for key, value in index_of_pers.items():
        if type(value) == int:
            count_pers[key] = matrix_freq[value]
        else:
            count_pers[key] = 0
    pers = {'Моника': count_pers['Моника'] + count_pers['Мон'],
            'Рэйчел': count_pers['Рэйчел'] + count_pers['Рейч'],
            'Чендлер': count_pers['Чендлер'] + count_pers['Чэндлер'] + count_pers['Чен'],
            'Фиби': count_pers['Фиби'] + count_pers['Фибс'],
            'Джоуи': count_pers['Джоуи'] + count_pers['Джои'] + count_pers['Джо'],
            'Росс': count_pers['Росс']}
    max_pers = max(pers.values())
    max_dict = {key: value for key, value in pers.items() if value == max_pers}
    return ','.join(list(max_dict.keys()))