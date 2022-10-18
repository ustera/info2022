import streamlit as st
from import_data import import_files, import_bert
from request_processing import request_search_bert, create_output, search_similar, request_search_bm25, request_search_tfidf
import time
import base64


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def search():
    st.set_page_config(page_title="Поисковик", page_icon=":love_letter:", layout="centered")
    texts, vectorizer, bert_matrix_answers, matrix_bm25, morph, stoplist, matrix_tfidf, vectorizer_tfidf = import_files()
    tokenizer, model = import_bert()
    st.title(":love_letter: (Не) очень романтичный поисковик")
    set_bg('bg.png')
    with st.expander("Как пользоваться"):
        st.write("""
            1. Выберите метод поиска в выпадающем меню: **TF-IDF**, **BM25** или **BERT**
            2. На слайдере переместите точку на число с нужным количеством ответов в выдаче
            3. Введите свой запрос
            4. Нажмите "Поиск"
        """)
    form = st.form(key="annotation")
    with form:
        cols = st.columns((1, 1))
        method = cols[0].selectbox('Метод поиска', ('TF-IDF', 'BM25', 'BERT'))
        search_count = cols[1].slider('Количество ответов', min_value=1, max_value=100)
        request = st.text_input('Поле для запроса')
        submitted = st.form_submit_button(label="Поиск")
    if submitted:
        with st.spinner('Поиск лучших вариантов'):
            start = time.time()
            if request.isspace() or request == '':
                st.write('Некорректный запрос. Не очень продуктивно искать пустые строки :(')
            else:
                if method == 'BERT':
                    vec_request_bert = request_search_bert(tokenizer, model, request)
                    sim_bert = search_similar(bert_matrix_answers, vec_request_bert)
                    result = create_output(sim_bert, list(texts.values()))
                    end = time.time()
                    for res in result[:search_count]:
                        st.write('* ' + res)
                if method == 'BM25':
                    vec_request_bm25 = request_search_bm25(vectorizer, request, morph, stoplist)
                    sim_bm25 = search_similar(matrix_bm25, vec_request_bm25)
                    result = create_output(sim_bm25.toarray(), list(texts.values()))
                    end = time.time()
                    for res in result[:search_count]:
                        st.write('* ' + res)
                if method == 'TF-IDF':
                    vec_request_tfidf = request_search_tfidf(vectorizer_tfidf, request, morph, stoplist)
                    sim_tfidf = search_similar(matrix_tfidf, vec_request_tfidf)
                    result = create_output(sim_tfidf.toarray(), list(texts.values()))
                    end = time.time()
                    for res in result[:search_count]:
                        st.write('* ' + res)
                total_time = round((end - start) * 1000)
                st.write(f"<p style='opacity:.5'>Поиск занял {total_time} мс</p>", unsafe_allow_html=True)


if __name__ == '__main__':
    search()
