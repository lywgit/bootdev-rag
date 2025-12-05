import mimetypes
from google import genai 
from google.genai.types import GenerateContentResponse
from textwrap import dedent
from .gemini_util import (
    get_gemini_client,
    DEFAULT_GEMINI_MODEL
)

def llm_describe_image(query:str, image_data:bytes, mime:str, model=DEFAULT_GEMINI_MODEL) -> GenerateContentResponse:
    """Rewrite the input query based on the image content to improve search results."""
    client = get_gemini_client()
    prompt = dedent("""
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
        """).strip()
    parts = [
        prompt,
        genai.types.Part.from_bytes(data=image_data, mime_type=mime),
        query.strip()
    ]
    response = client.models.generate_content(model=model, contents=parts)
    return response

def describe_image_command(query:str, image_path:str) -> GenerateContentResponse:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return llm_describe_image(query, image_data, mime)
