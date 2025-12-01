import json

MOVIE_DATA_PATH = 'data/movies.json'
STOPWORD_DATA_PATH = 'data/stopwords.txt'
CACHE_DIR = 'cache'
BM25_K1 = 1.5
BM25_B = 0.75

def load_movie_list() -> list:
    with open(MOVIE_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data['movies']

def load_stopwords() -> list:
    with open(STOPWORD_DATA_PATH, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words
