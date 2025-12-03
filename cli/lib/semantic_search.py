import os
import re
import numpy as np
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from .search_util import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    SCORE_PRECISION,
    DOCUMENT_PREVIEW_LENGTH,
    load_movie_list
)


class SemanticSearch:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2" ) -> None:
        self.model = SentenceTransformer(model_name)
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings:NDArray | None = None
        self.chunk_metadata:list[dict] | None = None
        self.chunk_embedding_file_path = os.path.join(CACHE_DIR,"chunk_embeddings.npy")
        self.chunk_metadata_file_path = os.path.join(CACHE_DIR,"chunk_metadata.json")

    def build_chunk_embeddings(self, documents:list[dict]) -> NDArray:
        self.documents = documents
        for doc in documents: # keys: id, title description
            self.document_map[doc['id']] = doc
        all_chunks:list[str] = []
        chunk_metadata:list[dict] = []
        for doc_idx, doc in enumerate(documents):
            if not doc["description"].strip():
                continue
            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = {
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                }
                chunk_metadata.append(meta)
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata
        with open(self.chunk_embedding_file_path, 'wb') as f:
            np.save(f, self.chunk_embeddings)
        with open(self.chunk_metadata_file_path, 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> NDArray:
        self.documents = documents
        for doc in documents: # keys: id, title description
            self.document_map[doc['id']] = doc
        # load if exists and verify length
        if os.path.exists(self.chunk_embedding_file_path) and os.path.exists(self.chunk_metadata_file_path):
            with open(self.chunk_embedding_file_path, 'rb') as f:
                loaded_chunk_embeddings:NDArray = np.load(f)
            with open(self.chunk_metadata_file_path, 'r') as f:
                loaded_chunk_metadata = json.load(f)['chunks']
            self.chunk_embeddings = loaded_chunk_embeddings
            self.chunk_metadata = loaded_chunk_metadata
            print(f"{len(loaded_chunk_embeddings)} Chunk embeddings loaded")
            return self.chunk_embeddings
            
        # create if not already exist
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query:str, limit:int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None or self.documents is None:
            raise ValueError("Use load_or_create_chunk_embeddings to generate embedding first")
        
        q_embedding = self.generate_embedding(query)
        chunk_scores:list[dict] = []
        for i_chunk, embedding in enumerate(self.chunk_embeddings):
            sim_score = cosine_similarity(q_embedding, embedding)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[i_chunk]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i_chunk]["movie_idx"],
                "score":sim_score
            })
        movie_scores:dict[int,float] = defaultdict(int)
        for chunk_score in chunk_scores:
            movie_idx =  chunk_score["movie_idx"]
            if chunk_score["score"] > movie_scores[movie_idx]:
                movie_scores[movie_idx] = chunk_score["score"]
        movie_score_list = [(movie, score) for movie, score in movie_scores.items()]
        movie_score_list.sort(key=lambda x: x[1], reverse=True)
        result = []
        for (movie_idx, score) in movie_score_list[:limit]:
            movie = self.documents[movie_idx]
            result.append({
                "id": movie['id'],
                "title": movie['title'],
                "document": movie['description'][:DOCUMENT_PREVIEW_LENGTH],
                "score": round(score, SCORE_PRECISION),
                "metadata": {}
            })
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

def chunk_command(text:str, chunk_size:int = DEFAULT_CHUNK_SIZE, overlap:int = DEFAULT_CHUNK_OVERLAP):
    words = [w for w in text.split() if w]
    n_chars = len(" ".join(words))
    print(f"Chunking {n_chars} characters")
    start_idx = 0
    i = 1
    while start_idx < len(words):
        print(f"{i}. {' '.join(words[start_idx:start_idx+chunk_size])}")
        # explicit exit when all words are covered to avoid last small chunk
        if start_idx + chunk_size >= len(words):
            break
        start_idx += chunk_size - overlap
        i += 1

def semantic_chunk(text:str, max_chunk_size:int = 4, overlap:int = 0) -> list[str]:
    text = text.strip()
    if text == "":
        return []
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    # if there's only one sentence and it doesn't end with a punctuation mark like ., !, or ?, treat the whole text as one sentence.
    if len(sentences) == 1 and not sentences[0].endswith(('.','!','?')):
        return sentences

    start_idx = 0
    chunks = []
    while start_idx < len(sentences):
        sentence = ' '.join(sentences[start_idx:start_idx+max_chunk_size]).strip()
        if not sentence:
            continue
        chunks.append(sentence)
        # explicit exit when all words are covered to avoid last small chunk
        if start_idx + max_chunk_size >= len(sentences):
            break
        start_idx += max_chunk_size - overlap
    return chunks


def semantic_chunk_command(text:str, max_chunk_size:int = 4, overlap:int = 0):
    n_chars = len(text)
    print(f"Semantically chunking {n_chars} characters")
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    for i, sentence in enumerate(chunks):
        print(f"{i+1}. {sentence}")

def embed_chunk_command():
    docs = load_movie_list()
    css = ChunkedSemanticSearch()
    print("Generating chunk embeddings")
    embeddings = css.load_or_create_chunk_embeddings(docs)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunk_command(query:str, limit:int = DEFAULT_SEARCH_LIMIT):
    docs = load_movie_list()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(docs)
    search_results = css.search_chunks(query, limit)
    for i, result in enumerate(search_results):
        print(f"\n{i+1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")