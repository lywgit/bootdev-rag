import json

MOVIE_DATA_PATH = 'data/movies.json'
STOPWORD_DATA_PATH = 'data/stopwords.txt'
CACHE_DIR = 'cache'

def load_movie_list() -> list:
    with open(MOVIE_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data['movies']

def load_stopwords() -> list:
    with open(STOPWORD_DATA_PATH, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words
