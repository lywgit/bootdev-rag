import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_util import load_movie_list

DEFAULT_CLIP_MODEL = "clip-ViT-B-32"

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class MultimodalSearch:
    def __init__(self, docs:list[dict], model_name:str = DEFAULT_CLIP_MODEL) -> None:
        self.model:SentenceTransformer = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
        self.embedding = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path:str) -> NDArray:
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0] # type: ignore
        return embedding

    def search_with_image(self, image_path:str, limit:int = 5) -> list[dict]:
        query_embedding = self.embed_image(image_path)
        scores = []
        for embedding in self.embedding:
            score = cosine_similarity(query_embedding, embedding)
            scores.append(score)
        sorted_idx = sorted([idx for idx in range(len(scores))], key=lambda x: scores[x], reverse=True)
        res = []
        for idx in sorted_idx[:limit]:
            doc = self.docs[idx]
            res.append({
                "doc_id":doc["id"],
                "title":doc["title"],
                "description":doc["description"],
                "score":scores[idx]
            })
        return res

def image_search_command(image_path:str, model_name:str = DEFAULT_CLIP_MODEL):
    docs = load_movie_list()
    ms = MultimodalSearch(docs, model_name)
    return ms.search_with_image(image_path)

    

def verify_image_embedding(image_path:str):
    docs = load_movie_list()
    ms = MultimodalSearch(docs[:1]) # doc is not needed here 
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
