def find_max(matrix_freq, dictionary):
    """ Finds the maximum frequency and returns lemma with this frequency """
    max_count = max(matrix_freq)
    for i in range(0, len(matrix_freq)):
        if matrix_freq[i] == max_count:
            ind = i
    return 'а) Самое частотное слово: ' + dictionary[ind]


def find_min(matrix_freq, dictionary):
    """ Finds the minimum frequency and returns lemmas with this frequency """
    min_count = min(matrix_freq)
    min_ind = []
    rare_list = []
    for i in range(0, len(matrix_freq)):
        if matrix_freq[i] == min_count:
            min_ind.append(i)
    for ind in min_ind:
        rare_list.append(dictionary[ind])
    return rare_list