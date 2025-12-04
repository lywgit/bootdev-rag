import argparse
from lib.hybrid_search import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_HYBRID_RRF_K,
    normalize_command,
    weighted_search_command,
    rrf_search_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # -- commands --
    # normalize
    normalize_parser = subparsers.add_parser("normalize", description="Min max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="scores to normalize")
    # weighted-search 
    ws_parser = subparsers.add_parser("weighted-search", description="Hybrid search of bm25 and semantic")
    ws_parser.add_argument("query", type=str, help="Query string")
    ws_parser.add_argument("--alpha", type=float, help="Weighting parameter. 1.0=bm25, 0.0=semantic. Default=0.5", default=0.5)
    ws_parser.add_argument("--limit", type=int, help=f"Search result limit. Default {DEFAULT_SEARCH_LIMIT}", default=DEFAULT_SEARCH_LIMIT)
    # rrf-search
    rrf_parser = subparsers.add_parser("rrf-search", description="Hybrid search using bm25 and semantic Reciprocal Rank Fusion ")
    rrf_parser.add_argument("query", type=str, help="Query string")
    rrf_parser.add_argument("--k", type=float, help=f"RRF Weighting parameter. Default={DEFAULT_HYBRID_RRF_K}", default=DEFAULT_HYBRID_RRF_K)
    rrf_parser.add_argument("--limit", type=int, help=f"Search result limit, Default {DEFAULT_SEARCH_LIMIT}", default=DEFAULT_SEARCH_LIMIT)
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite","expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch","cross_encoder"], help="Query re-ranking method", default=None)

    
    # -- parse and run --
    args = parser.parse_args()
    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()