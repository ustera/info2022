from import_data import read_file
from preprocessing import create_dict
from create_matrix import indexing
from request_processing import request_search, create_output, search_similar
import argparse


def main(path, request):
    """ Main def for the tool. Combines other defs into easy-to-use pipelines """
    texts = read_file(path)
    parsed_texts = create_dict(texts)
    index_matrix, vectorizer = indexing(parsed_texts.values())
    while request != ' ':
        vec_request = request_search(vectorizer, request)
        sim = search_similar(index_matrix, vec_request)
        result = create_output(sim, list(texts.values()))
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


