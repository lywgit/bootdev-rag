import json

MOVIE_DATA_PATH = 'data/movies.json'
STOPWORD_DATA_PATH = 'data/stopwords.txt'
CACHE_DIR = 'cache'
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0 # typically 20% overlap
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 10
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_HYBRID_ALPHA = 0.5 
DEFAULT_HYBRID_RRF_K = 60

def load_movie_list() -> list:
    with open(MOVIE_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data['movies']

def load_stopwords() -> list:
    with open(STOPWORD_DATA_PATH, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words
