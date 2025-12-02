import os
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from .search_util import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movie_list
)


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings:NDArray | None = None  
        self.documents:list[dict] | None = None 
        self.document_map:dict[int,dict] = dict()
        self.embedding_file_path = os.path.join(CACHE_DIR, 'movie_embeddings.npy')

    @property 
    def total_document(self):
        if self.documents is None:
            return 0
        return len(self.documents)

    def generate_embedding(self, text:str) -> NDArray:
        if len(text) == 0:
            raise ValueError("Text can not be empty")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embedding(self, documents:list[dict]):
        self.documents = documents
        document_strs = []
        for doc in documents: # keys: id, title description
            self.document_map[doc['id']] = doc
            document_strs.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(document_strs, show_progress_bar=True) 
        with open(self.embedding_file_path, 'wb') as f:
            np.save(f, self.embeddings)
            
    def load_or_create_embeddings(self, documents:list[dict]) -> NDArray:
        self.documents = documents
        for doc in documents: # keys: id, title description
            self.document_map[doc['id']] = doc
        # load if exists and verify length
        if os.path.exists(self.embedding_file_path):
            with open(self.embedding_file_path, 'rb') as f:
                loaded_embeddings:NDArray = np.load(f)
            if len(loaded_embeddings) == len(documents):
                self.embeddings = loaded_embeddings
                return self.embeddings
        # create
        assert self.embeddings is not None # just for type checker 
        self.build_embedding(documents)
        return self.embeddings

    def search(self, query:str, limit:int = DEFAULT_SEARCH_LIMIT) -> list:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embedding = self.generate_embedding(query)
        doc_scores = []
        for i in range(self.total_document):
            score = cosine_similarity(q_embedding, self.embeddings[i])
            doc_scores.append((score, self.documents[i]))
        doc_scores.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in doc_scores[:limit]:
            result.append(({
                "score":score,
                "title":doc["title"],
                "description":doc["description"]
            }))
        return result


def embed_query_text(query:str) -> NDArray:
    ss = SemanticSearch() 
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text:str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movie_list()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query:str, limit:int = DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    docs = load_movie_list()
    ss.load_or_create_embeddings(docs)
    results = ss.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]} (score: {result["score"]:.4f}) ")
        print(f"  {result["description"]}" )
    return results

