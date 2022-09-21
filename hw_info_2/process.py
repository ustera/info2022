from import_data import folder_pass
from preprocessing import create_dict
from create_matrix import indexing
from request_processing import request_search, corpus_similar
import argparse


def main(path, request):
	# тут была переменная mappe, я не знаю, почему она так обозвана, это уже традиция с тех пор,
	# как я начинала работать с файлами и путями. убрала, чтобы не смущало
	""" Main def for the tool. Combines other defs into easy-to-use pipelines """
	texts = folder_pass(path)
	parsed_texts = create_dict(texts)
	index_matrix, dictionary, vectorizer = indexing(parsed_texts.values())
	while request != ' ':
		vec_request = request_search(vectorizer, request)
		corpus_similar(parsed_texts, index_matrix, vec_request)
		request = input('Введите следующий запрос. Для выхода введите пробел. ')


if __name__ == '__main__':
	""" Gets argument for work with folder and gets the first request """
	argparser = argparse.ArgumentParser()
	argparser.add_argument("path", type=str, help="Путь до папки")
	argparser.add_argument("request", type=str, help="Первый запрос")
	args = argparser.parse_args()
	main(path=args.path, request=args.request)


