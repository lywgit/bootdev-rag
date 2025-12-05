
import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # rag
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # summarize
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + generate summary from answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarize")
    summarize_parser.add_argument("--limit",type=int, help="Number of results", default=5)

    # citations
    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + generate answer with citations)"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit",type=int, help="Number of results", default=5)

    # question
    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + generate answer with citations)"
    )
    question_parser.add_argument("question", type=str, help="Question")
    question_parser.add_argument("--limit",type=int, help="Number of results", default=5)

    # parse and run
    args = parser.parse_args()
    match args.command:
        case "rag":
            query = args.query
            res = rag_command(query)
            res_docs = res['search_results']
            rag_response = res['rag_answer']
            print("Search Results:")
            for doc in res_docs:
                print(f"  - {doc["title"]}")
            print()
            print("RAG Response:")
            print(rag_response)
        case "summarize":
            res = summarize_command(args.query, args.limit)
            res_docs = res['search_results']
            rag_summary = res['rag_summary']
            print("Search Results:")
            for doc in res_docs:
                print(f"  - {doc["title"]}")
            print()
            print("LLM Summary:")
            print(rag_summary)
        case "citations":
            res = citations_command(args.query, args.limit)
            res_docs = res['search_results']
            rag_response = res['rag_citations']
            print("Search Results:")
            for doc in res_docs: 
                print(f"  - {doc["title"]}")
            print()
            print("LLM Answer:")
            print(rag_response)
        case "question":
            res = question_command(args.question, args.limit)
            res_docs = res['search_results']
            rag_response = res['rag_answer']
            print("Search Results:")
            for doc in res_docs: 
                print(f"  - {doc["title"]}")
            print()
            print("Answer:")
            print(rag_response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()