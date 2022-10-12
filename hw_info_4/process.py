from import_data import read_file
from create_matrix_bm25 import indexing_bm25
from create_matrix_bert import indexing_bert
from request_processing import request_search_bert, create_output, search_similar
from score import count_score
import argparse


def main(path, request):
    """ Main def for the tool. Combines other defs into easy-to-use pipelines """
    texts = read_file(path)
    index_matrix_answer, vectorizer = indexing_bm25(texts.values())
    index_matrix_question = vectorizer.transform(texts.keys())
    bert_matrix_questions, tokenizer, model = indexing_bert(texts.keys())
    bert_matrix_answers, tokenizer, model = indexing_bert(texts.values())
    print('score bm25: ' + str(count_score(index_matrix_question, index_matrix_answer)))
    print('score bert: ' + str(count_score(bert_matrix_questions, bert_matrix_answers)))
    while request != ' ':
        vec_request_bert = request_search_bert(tokenizer, model, request)
        sim_bert = search_similar(bert_matrix_answers, vec_request_bert)
        print('Наиболее подходящие 5 ответов:')
        result = create_output(sim_bert, list(texts.values()))
        for res in result[:5]:
            print(res)

        request = input('Введите следующий запрос. Для выхода введите пробел. ')


if __name__ == '__main__':
    """ Gets argument for work with folder and gets the first request """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="Путь до папки")
    argparser.add_argument("request", type=str, help="Первый запрос")
    args = argparser.parse_args()
    main(path=args.path, request=args.request)


