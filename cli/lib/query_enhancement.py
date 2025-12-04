from .gemini_util import (
    DEFAULT_GEMINI_MODEL,
    get_gemini_client
)

def llm_spell_check(query:str, model:str=DEFAULT_GEMINI_MODEL) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(model=model,contents=f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:""")
    return response.text if response.text else ""

def llm_rewrite(query:str, model:str=DEFAULT_GEMINI_MODEL) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(model=model,
                                              contents=f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:""")
    return response.text if response.text else ""

def llm_expand(query:str, model:str=DEFAULT_GEMINI_MODEL) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(model=model,
                                              contents=f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
""")
    return response.text if response.text else ""


def enhance_query(query:str, method:str|None = None) -> str:
    if method is None:
        return query
    orig_query = query
    match method:
        case "spell":
            query = llm_spell_check(query)
            print(f"Enhanced query ({method}): '{orig_query}' -> '{query}'\n")
        case "rewrite":
            query = llm_rewrite(query)
            print(f"Enhanced query ({method}): '{orig_query}' -> '{query}'\n")
        case "expand":
            query = llm_expand(query)
            print(f"Enhanced query ({method}): '{orig_query}' -> '{query}'\n")
        case _:
            raise ValueError(f"Unexpected method: {method}")
    return query

