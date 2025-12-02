#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    DEFAULT_SEARCH_LIMIT
    
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", description="Available commands")
    # -- commands
    # verify 
    subparsers.add_parser("verify", help="Print model information")
    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Print model information")
    embed_text_parser.add_argument("text", type=str, help= "Text to embed")
    # verify_embeddings
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    # embedquery
    eq_parser = subparsers.add_parser("embedquery", help="Get embedding of query text")
    eq_parser.add_argument("query", type=str, help="The query text to embed")
    # search
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="Query string")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                               help="Limit on returned result")

    # -- Parse and run
    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            results = search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()