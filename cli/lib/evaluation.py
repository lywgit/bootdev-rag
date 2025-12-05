import json
from textwrap import dedent
from .gemini_util import (
    DEFAULT_GEMINI_MODEL,
    get_gemini_client
)

def llm_evaluate(query:str, formatted_results:list[str], model:str=DEFAULT_GEMINI_MODEL) -> list[int]:
    client = get_gemini_client()
    prompt = dedent(f"""
        Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {"\n".join(formatted_results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]
        """).strip()
    response = client.models.generate_content(model=model, contents=prompt, 
        config={"response_mime_type": "application/json"})
    response_text = response.text or ""
    scores = [int(score) for score in json.loads(response_text)]
    return scores


def evaluate_relevance(query:str, formatted_results:list[str]) -> list[int]:
    return llm_evaluate(query, formatted_results)