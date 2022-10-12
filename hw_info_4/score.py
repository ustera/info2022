import numpy as np
from tqdm import tqdm


def count_score(matrix_questions, matrix_answers):
    """ Counts scores for questions and answers """
    sim_m = np.dot(matrix_answers, matrix_questions.T)
    in_top_5 = 0
    for i in tqdm(range(0, sim_m.shape[0])):
        sorted_m = np.argsort(sim_m[i], axis=0)[::-1]
        if i in sorted_m[:5]:
            in_top_5 += 1
    return in_top_5 / matrix_questions.shape[0]
