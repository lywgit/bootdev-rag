import string
import os
import math
import pickle
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer 

from .search_util import (
    load_movie_list, 
    load_stopwords,
    CACHE_DIR,
    BM25_K1,
    BM25_B
)



def remove_punctuation(text:str) -> str:
    map_table = str.maketrans('', '', string.punctuation) # remove punctuation
    new_text = str.translate(text, map_table)
    return new_text

def preprocess_text(text:str) -> str:
    text = text.lower()
    text = remove_punctuation(text)
    return text

def word_tokenize(text:str) -> list[str]:
    words = [w for w in text.split() if w != ""]
    words = remove_stopwords(words)
    words = stem_words(words)
    return words

def stem_words(words:list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]

def remove_stopwords(words:list[str]):
    stop_words = load_stopwords()
    words = [word for word in words if word not in stop_words]
    return words
        

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

     
class InvertedIndex:
    def __init__(self):
        self.index: dict[str,set] = defaultdict(set) # token -> doc_ids
        self.docmap: dict[int,dict] = dict() # doc_id -> doc obj
        self.term_frequency: dict[int, Counter[str]] = defaultdict(Counter) # doc_id -> token -> cnt
        self.doc_lengths: dict[int,int] = dict()

    @property
    def total_document(self) -> int:
        return len(self.docmap)
    
    def __add_document(self, doc_id:int, text:str):
        tokens = word_tokenize(preprocess_text(text))
        # update inverted index and term frequencies
        for token in tokens: 
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1
        # update doc_lengths
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if self.total_document == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.total_document

    def _process_single_token_input(self, term:str) -> str:
        q_tokens = word_tokenize(preprocess_text(term))
        if len(q_tokens) != 1:
            raise Exception(f"Expect single token but got {len(q_tokens)}: {q_tokens}")
        q_token = q_tokens[0]
        return q_token

    def get_documents(self, term:str) -> list[int]: 
        # term is a index key
        doc_ids = list(self.index.get(term.lower(), set()))
        doc_ids.sort()
        return doc_ids
    
    def build(self):
        movies = load_movie_list() # keys = id, title, description
        for movie in movies:
            # a copy of original data
            self.docmap[movie['id']] = movie
            # update inverted index
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")


    def get_tf(self, doc_id:int, term:str) -> int:
        term_token = self._process_single_token_input(term)
        return self.term_frequency[doc_id][term_token]
    
    def get_idf(self, term:str) -> float:
        term_token = self._process_single_token_input(term)
        doc_count = self.total_document
        # print("doc_count",doc_count )
        term_doc_count = len(self.index[term_token])
        # print("term_doc_count", len(self.index[term_token]))
        return math.log( (doc_count+1) / (term_doc_count+1) )
    
    def get_tf_idf(self, doc_id:int, term:str) -> float:
        term_token = self._process_single_token_input(term)
        tf = self.get_tf(doc_id, term_token) 
        idf = self.get_idf(term_token)
        # print(f'-- tf:{tf} / idf:{idf}')
        return tf * idf
    
    def get_bm25_tf(self, doc_id:int, term:str, k1:float = BM25_K1, b:float = BM25_B) -> float:
        term_token = self._process_single_token_input(term)
        tf = self.get_tf(doc_id, term_token)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length() )
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) 

    def get_bm25_idf(self, term:str) -> float:
        term_token = self._process_single_token_input(term)
        doc_count = self.total_document
        term_doc_count = len(self.index[term_token])
        return math.log( (doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def bm25(self, doc_id:int, term:str, k1:float = BM25_K1, b:float = BM25_B):
        term_token = self._process_single_token_input(term)
        return self.get_bm25_tf(doc_id, term_token, k1, b) * self.get_bm25_idf(term_token)

    def bm25_search(self, query:str, limit:int, k1:float = BM25_K1, b:float = BM25_B) -> list[tuple]:
        q_tokens = word_tokenize(preprocess_text(query))
        doc_scores:dict[int,float] = defaultdict(float)
        for q_token in set(q_tokens):
            doc_ids = self.get_documents(q_token)
            for doc_id in doc_ids:
                doc_scores[doc_id] += self.bm25(doc_id, q_token, k1, b)
        top_doc_scores = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:limit]
        result = [(self.docmap[doc_id],score) for doc_id, score in top_doc_scores]
        return result

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, 'index.pkl'), 'wb') as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, 'docmap.pkl'), 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(CACHE_DIR, 'term_frequencies.pkl'), 'wb') as f:
            pickle.dump(self.term_frequency, f)
        with open(os.path.join(CACHE_DIR, 'doc_lengths.pkl'), 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def _load(self):
        with open(os.path.join(CACHE_DIR, 'index.pkl'), 'rb') as f:
            self.index = pickle.load(f)
        with open(os.path.join(CACHE_DIR, 'docmap.pkl'), 'rb') as f:
            self.docmap = pickle.load(f)
        with open(os.path.join(CACHE_DIR, 'term_frequencies.pkl'), 'rb') as f:
            self.term_frequency = pickle.load(f)
        with open(os.path.join(CACHE_DIR, 'doc_lengths.pkl'), 'rb') as f:
            self.doc_lengths = pickle.load(f)

    def load(self):
        try:
            self._load()
        except Exception as e :
            print("Fail to load inverted_index")
            raise e



def search_command(query:str, limit:int = 5):
    inverted_index = InvertedIndex()
    inverted_index.load()
    preprocessed_query = preprocess_text(query)
    query_tokens = word_tokenize(preprocessed_query)
    print("query_tokens", query_tokens)
    matches = set()
    for query_token in query_tokens:
        doc_ids = inverted_index.get_documents(query_token)
        for doc_id in doc_ids:
            matches.add(doc_id)
            if len(matches) >= limit:
                print("matches", matches)
                return [inverted_index.docmap[idx] for idx in sorted(matches)]
    
    return [inverted_index.docmap[idx] for idx in sorted(matches)]

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id:int, term:str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)

def idf_command(term:str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)
 
def tf_idf_command(doc_id:int, term:str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf_idf(doc_id, term)  
 
def bm25_tf_command(doc_id:int, term:str, k1:float = BM25_K1, b:float = BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term:str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)   

def bm25_search_command(query:str, limit:int = 5, k1:float = BM25_K1, b:float = BM25_B):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit, k1, b) 
