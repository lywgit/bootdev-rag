import json
from textwrap import dedent
from .gemini_util import (
    DEFAULT_GEMINI_MODEL,
    get_gemini_client
)
from .search_util import (
    load_movie_list, 
    load_golden_dataset
    )
from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch

def evaluate_rrf(limit:int):
    docs = load_movie_list()
    test_cases = load_golden_dataset()
    ss = SemanticSearch()
    ss.load_or_create_embeddings(docs)
    hs = HybridSearch(docs)
    print(f"k={limit}")
    print("")
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        total_relevant = len(relevant_docs)
        search_result = hs.rrf_search(query, k=60, limit=limit) # list of (doc_id, doc, bm25_rank, sem_rank, rrf_score)
        search_result_titles = [res[1]["title"] for res in search_result]
        total_retrieved = len(search_result)
        relevant_retrieved = len(set(relevant_docs) & set(search_result_titles))
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}" )
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(search_result_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


def llm_evaluate(query:str, formatted_results:list[str], model:str=DEFAULT_GEMINI_MODEL) -> list[int]:
    client = get_gemini_client()
    prompt = dedent(f"""
        Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {"\n".join(formatted_results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]
        """).strip()
    response = client.models.generate_content(model=model, contents=prompt, 
        config={"response_mime_type": "application/json"})
    response_text = response.text or ""
    scores = [int(score) for score in json.loads(response_text)]
    return scores


def evaluate_relevance(query:str, formatted_results:list[str]) -> list[int]:
    return llm_evaluate(query, formatted_results)