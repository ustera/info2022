from import_data import folder_pass
from preprocessing import create_dict
from create_matrix import indexing
from freq import find_max, find_min
from search import each_in_collection, most_popular_person
import argparse


def main(mappe):
	""" Main def for the tool. Combines other defs into easy-to-use pipelines """
	texts = folder_pass(mappe)
	parsed_texts = create_dict(texts)
	index_matrix, matrix_freq, dictionary, vectorizer = indexing(parsed_texts.values())
	print(find_max(matrix_freq, dictionary))
	print('Редких слов получилось очень много. Вот список:')
	print(find_min(matrix_freq, dictionary))
	print(each_in_collection(index_matrix, dictionary))
	print('Самый популярный герой: ' + most_popular_person(matrix_freq, vectorizer))


if __name__ == '__main__':
	""" Get argument for work with folder """
	argparser = argparse.ArgumentParser()
	argparser.add_argument("mappe", type=str, help="Путь до папки")
	args = argparser.parse_args()
	main(mappe=args.mappe)


