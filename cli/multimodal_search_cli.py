import argparse
from lib.search_util import DOCUMENT_PREVIEW_LENGTH
from lib.multimodal_search import (
    verify_image_embedding,
    image_search_command
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal search")
    subparser = parser.add_subparsers(dest="command", description="Available commands")
    # verify_image_embedding
    vie_parser = subparser.add_parser("verify_image_embedding", description="Verify image embedding")
    vie_parser.add_argument("image_path", type=str, help="Image path")
    # image_search
    imgs_parser = subparser.add_parser("image_search", description="Search with image")
    imgs_parser.add_argument("image_path", type=str, help="Image path")
    
    
    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]} (similarity: {res["score"]:.3f})")
                print(f"  {res["description"][:DOCUMENT_PREVIEW_LENGTH]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
