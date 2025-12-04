import os
from dotenv import load_dotenv
from google import genai

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-001"

def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise OSError("Environment variable 'GEMINI_API_KEY' is not set.")
    client = genai.Client(api_key=api_key)
    return client