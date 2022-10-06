import json
import random


def read_file(path):
    """ Gets the path to the .jsonl file. Returns a dictionary with question and answers from them """
    texts = {}
    with open(path, 'r') as f:
        json_list = list(f)
        for json_str in json_list:
            line = json.loads(json_str)
            max_rate = 0
            best_answer = ''
            for answer in line['answers']:
                if answer['author_rating']['value'] is '':
                    answer['author_rating']['value'] = 0
                if int(answer['author_rating']['value']) >= max_rate:
                    max_rate = int(answer['author_rating']['value'])
                    best_answer = answer['text']
            texts[line['question']] = best_answer
    keys = random.sample(texts.keys(), 50000)
    texts_sample = {k: texts[k] for k in keys}

    return texts_sample
