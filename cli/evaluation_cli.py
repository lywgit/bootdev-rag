import argparse
from lib.search_util import (
    load_movie_list, 
    load_golden_dataset
    )
from lib.hybrid_search import HybridSearch
from lib.semantic_search import SemanticSearch

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


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    evaluate_rrf(limit)


if __name__ == "__main__":
    main()