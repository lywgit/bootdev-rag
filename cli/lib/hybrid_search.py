import os
from collections import defaultdict
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_util import (
    load_movie_list,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_HYBRID_ALPHA,
    DOCUMENT_PREVIEW_LENGTH,
    DEFAULT_HYBRID_RRF_K,
    SEARCH_LIMIT_MULTIPLIER
)
from .query_enhancement import enhance_query
from .reranking import (
    rerank_individual,
    rerank_batch,
    rerank_cross_encoder
)
from .evaluation import evaluate_relevance

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.docmap = dict()
        for doc in documents:
            doc_id = doc["id"]
            self.docmap[doc_id] = doc
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit) 

    def weighted_search(self, query, alpha, limit=5) -> list[tuple]: # (doc_id, doc, bm25, sem, hybrid)
        source_limit = 500*limit 
        doc_ids = set()
        # bm25 scores
        bm25_result = self._bm25_search(query, source_limit) # list[(doc_id, doc, score)]
        normalized_bm25_score = normalize([res[2] for res in bm25_result])
        bm25_score_dic = defaultdict(float)
        for (doc_id, _, _), norm_bm25_score in zip(bm25_result, normalized_bm25_score):
            bm25_score_dic[doc_id] = norm_bm25_score
            doc_ids.add(doc_id)
        # semantic scores
        semantic_result = self.semantic_search.search_chunks(query, source_limit) # list[dict] / keys: id, title, document, score, metadata
        normalized_semantic_score = normalize([res["score"] for res in semantic_result])
        semantic_score_dic = defaultdict(float)
        for (res), norm_semantic_score in zip(semantic_result, normalized_semantic_score):
            doc_id = res["id"]
            semantic_score_dic[doc_id] = norm_semantic_score
            doc_ids.add(doc_id)
        # hybrid scores
        doc_scores:list[tuple] = list() # (doc_id, bm25_score, semantic_score, hybrid_score)
        for doc_id in doc_ids:
            h_score = hybrid_score(bm25_score_dic[doc_id], semantic_score_dic[doc_id], alpha)
            doc_scores.append((doc_id, bm25_score_dic[doc_id], semantic_score_dic[doc_id], h_score))
        doc_scores.sort(key=lambda x: x[3], reverse=True)
        # result
        result:list[tuple] = [] # list of (doc_id, doc, bm25, sem, hybrid)
        for doc_id, bm25, sem, h in doc_scores[:limit]:
            result.append((doc_id, self.docmap[doc_id], bm25, sem, h))
        return result

    def rrf_search(self, query:str, k:float = DEFAULT_HYBRID_RRF_K, limit:int = 10) -> list[tuple]:
        source_limit = 500*limit 
        doc_score_dic = defaultdict(float)
        
        bm25_result = self._bm25_search(query, source_limit) # list[(doc_id, doc, score)]
        bm25_rank_dic = dict()
        for i, (doc_id, _, _) in enumerate(bm25_result):
            rank = i+1
            doc_score_dic[doc_id] += rrf_score(rank, k)
            bm25_rank_dic[doc_id] = rank

        semantic_result = self.semantic_search.search_chunks(query, source_limit) # list[dict] / keys: id, title, document, score, metadata
        semantic_rank_dic = dict()
        for i, res in enumerate(semantic_result):
            rank = i+1 
            doc_id = res["id"]   
            doc_score_dic[doc_id] += rrf_score(rank, k)
            semantic_rank_dic[doc_id] = rank

        doc_scores:list[tuple] = [(doc_id, score) for doc_id, score in doc_score_dic.items()] # (doc_id, score)
        doc_scores.sort(key=lambda x:x[1], reverse=True)

        result:list[tuple] = [] # list of (doc_id, doc, bm25_rank, sem_rank, rrf_score)
        for doc_id, score in doc_scores[:limit]:
            result.append((doc_id, self.docmap[doc_id], bm25_rank_dic.get(doc_id, -1), semantic_rank_dic.get(doc_id, -1), score))
        return result


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k:float = DEFAULT_HYBRID_RRF_K):
    return 1 / (k + rank)

def normalize(scores:list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0]*len(scores)
    return [(s-min_score)/(max_score-min_score) for s in scores]

def normalize_command(scores:list[float]):
    normalized_score = normalize(scores)
    for score in normalized_score:
        print(f"* {score:.4f}")

def weighted_search_command(query:str, alpha:float = DEFAULT_HYBRID_ALPHA, limit:int = DEFAULT_SEARCH_LIMIT):
    docs = load_movie_list()
    hs = HybridSearch(docs)
    res = hs.weighted_search(query, alpha, limit)
    for i, (_, doc, bm25, sem, hybrid) in enumerate(res):
        print(f"{i+1}. {doc["title"]}")
        print(f"  Hybrid Score: {hybrid:.4f}")
        print(f"  BM25: {bm25:.4f}, Semantic: {sem:.4f}")
        print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")


def rrf_search_command(query:str, k:float=DEFAULT_HYBRID_RRF_K, limit:int = DEFAULT_SEARCH_LIMIT, enhance_method:str|None = None,
                       rerank_method:str|None = None, evaluate:bool = False):
    query = enhance_query(query, enhance_method)
    docs = load_movie_list()
    hs = HybridSearch(docs)
    res_with_score: list[tuple] = []
    if rerank_method is None:
        res = hs.rrf_search(query, k, limit)
        for i, (_, doc, bm25_rank, sem_rank, score) in enumerate(res):
            print(f"{i+1}. {doc["title"]}")
            print(f"  RRF Score: {score:.4f}")
            print(f"  BM25 Rank: {bm25_rank}, Semantic: {sem_rank}")
            print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")
        res_with_score = [(r, 0) for r in res ] # a dirty solution to make all res_with_score variable consistent
    else:
        print("-- Logging RRF result before reranking")
        search_limit = limit * SEARCH_LIMIT_MULTIPLIER
        res = hs.rrf_search(query, k, search_limit)  
        # log original search result for debug
        for i, (_, doc, bm25_rank, sem_rank, score) in enumerate(res):
            print(f"{i+1}. {doc["title"]}")
            print(f"  RRF Score: {score:.4f}")
            print(f"  BM25 Rank: {bm25_rank}, Semantic: {sem_rank}")
            print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")

        print(f"Reranking top {limit} results using {rerank_method} method...")
        if rerank_method == "individual":     
            res_with_score = rerank_individual(query, res, sleep_interval=0)
            for i, (res, rerank_score) in enumerate(res_with_score[:limit]):
                (_, doc, bm25_rank, sem_rank, score) = res
                print(f"{i+1}. {doc["title"]}")
                print(f"  Rerank Score: {rerank_score:.3f}/10")
                print(f"  RRF Score: {score:.4f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic: {sem_rank}")
                print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")
        elif rerank_method == "batch":
            res_with_rank = rerank_batch(query, res)
            for i, (res, rerank_rank) in enumerate(res_with_rank[:limit]):
                (_, doc, bm25_rank, sem_rank, score) = res
                print(f"{i+1}. {doc["title"]}")
                print(f"  Rerank Rank: {rerank_rank}")
                print(f"  RRF Score: {score:.4f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic: {sem_rank}")
                print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")
        elif rerank_method == "cross_encoder":
            res_with_score = rerank_cross_encoder(query, res)
            for i, (res, rerank_score) in enumerate(res_with_score[:limit]):
                (_, doc, bm25_rank, sem_rank, score) = res
                print(f"{i+1}. {doc["title"]}")
                print(f"  Cross Encoder Score: {rerank_score}")
                print(f"  RRF Score: {score:.4f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic: {sem_rank}")
                print(f"  {doc["description"][:DOCUMENT_PREVIEW_LENGTH]}")

    if evaluate:
        docs = [r[0][1] for r in res_with_score] 
        formatted_results = [f"{doc['title']} - {doc["description"]}" for doc in docs]
        scores = evaluate_relevance(query, formatted_results)
        print("LLM evaluation report:")
        for i, (doc, score) in enumerate(zip(docs, scores)):
            print(f"{i+1}. {doc['title']}: {score}/3")

