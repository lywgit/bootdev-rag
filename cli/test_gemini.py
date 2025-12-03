import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
assert api_key is not None
print(f"Using key {api_key[:6]}...")

from google import genai

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash-001", 
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    )
print(response.text)
assert response.usage_metadata is not None
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")