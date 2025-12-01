#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command, 
    build_command,
    tf_command,
    idf_command,
    tf_idf_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    # build command
    subparsers.add_parser("build", help="Build Inverted index and save to disk")
    # tf command
    tf_parser = subparsers.add_parser("tf", help="Print tf (term frequency) of given doc_id + term.")
    tf_parser.add_argument("doc_id", type=int, help='Document ID to search in')
    tf_parser.add_argument("term", type=str, help='The term to show the frequency')
    # idf command
    idf_parser = subparsers.add_parser("idf", help="Print idf (inverse document frequency) of a term")
    idf_parser.add_argument("term", help="The term to show idf")
    # tfidf command
    tfidf_parser = subparsers.add_parser("tfidf", help="Print tf-idf (term frequency - inverse document frequency) of doc_id + term")
    tfidf_parser.add_argument("doc_id", type=int, help='Document ID')
    tfidf_parser.add_argument("term", type=str, help='The term to show tf-idf')

    args = parser.parse_args()
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            result = search_command(args.query, limit=5)
            print(f'Found {len(result)} matches')
            for i, movie in enumerate(result):
                print(f"{i+1}. {movie['title']}")
        case "build":
            print("Building...")   
            build_command()        
            print("Build success")   
        case "tf":
            term_freq = tf_command(args.doc_id, args.term)
            print(term_freq) 
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()