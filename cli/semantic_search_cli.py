#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_command,
    embed_chunk_command,
    search_chunk_command,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
    
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
    # chunk (fixed size)
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text by chunk-size")
    chunk_parser.add_argument("text",type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="Chunk size by words", default=DEFAULT_CHUNK_SIZE)
    chunk_parser.add_argument("--overlap", type=int, help="Chunk overlap by words", default=DEFAULT_CHUNK_OVERLAP)

    # semantic_chunk 
    sem_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk text")
    sem_chunk_parser.add_argument("text",type=str, help="Text to chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, help="Max chunk size by sentences", default=4)
    sem_chunk_parser.add_argument("--overlap", type=int, help="Chunk overlap by sentences", default=0)

    # embed_chunks
    subparsers.add_parser("embed_chunks", help="Semantic chunk text")

    # search_chunked
    sc_parser = subparsers.add_parser("search_chunked", help="Semantic search with chunking")
    sc_parser.add_argument("query", type=str, help="Query string")
    sc_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
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
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunk_command()
        case "search_chunked":
            search_chunk_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()