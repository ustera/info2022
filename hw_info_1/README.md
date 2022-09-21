# HW1

## Запуск 
python3 process.py \[*путь до папки с файлами*]

## Файлы
`process.py` - основной файл, в котором происходит вызов функций проекта

`import_data.py` - файл, содержащий функции для обработки текста

`create_matrix.py` - файл, содержащий функции для индексирования

`freq.py` и `search.py` - файлы, использующийся для поиска ответов на задания

## Функции
`main()` - основная функция, в которой вызываются остальные ниже

`folder_pass(mappe)` - функция получает на вход путь до папки с файлами и возвращает словарь вида **{файл: текст}**

`create_dict(texts)` - функция получает на вход словарь **{файл: текст}**, для каждого текста вызывается функция `lemmatize()`

`lemmatize(text)` - функция получает на вход текст, очищает его, с помощью `pymorphy2` определяются леммы, возвращается строка из лемм

`indexing(texts)` - функция получает на вход тексты, инициализирует `CountVectorizer`, считает индексы. Возвращает матрицу, словарь, количественные вхождения и сам vectorizer

`find_max(matrix_freq, dictionary)` - функция находит среди количественных вхождений минимальное вхождени, затем создается список для хранения слов, так как их много, по словарю через индекс находятся сами слова

`find_min(matrix_freq, dictionary)` - функция аналогична предыдущей, только не создается список, так как слово единственное в данном случае 

`each_in_collection(index_matrix, dictionary)` - функция находит слова, которые есть в каждом документе коллекции. Для этого матрица "схлапывается в строку" (суммируются все строки) и находятся все слова с суммой равной 165 (количеству документов в коллекции)

`most_popular_person(matrix_freq, vectorizer)` - функция создает словарь с персонажами, для каждого персонажа находится количественное вхождение и суммируется по вариантам имен. Возращается имя персонажа с максимальным числом