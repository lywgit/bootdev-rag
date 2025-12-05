import argparse
from lib.multimodal import describe_image_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search")
    parser.add_argument("--image", type=str, help="Path to an input image", required=True)
    parser.add_argument("--query", type=str, help="Query string", required=True)

    args = parser.parse_args()
    response = describe_image_command(args.query, args.image) 

    print(f"Rewritten query: {response.text.strip() if response.text else ""}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")




if __name__ == '__main__':
    main()