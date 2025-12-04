import time
import json
from sentence_transformers import CrossEncoder
from textwrap import dedent
from .gemini_util import (
    DEFAULT_GEMINI_MODEL,
    get_gemini_client
)


def llm_rerank_single(query:str, doc:dict, model:str=DEFAULT_GEMINI_MODEL) -> float:
    """
    Args:
        doc: dict with two keys: "title", "document"
    """
    client = get_gemini_client()
    response = client.models.generate_content(model=model, contents=dedent(f"""
        Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:
        """).strip()
        )
    response_text = response.text or ""
    rerank_score = float(response_text.strip())
    return rerank_score


def rerank_individual(query:str, search_result:list[tuple], sleep_interval:int = 0) -> list[tuple[dict,float]]:
    new_result = []
    for res in search_result: # (score, doc, ...)
        doc = res[1]
        rerank_score = llm_rerank_single(query, {"title":doc["title"], "document":doc["description"]})
        time.sleep(sleep_interval) # avoid rate limit issue 
        new_result.append((res, rerank_score))
    new_result.sort(key=lambda x: x[1], reverse=True)
    return new_result


def llm_rerank_batch(query:str, doc_list_str:str, model:str=DEFAULT_GEMINI_MODEL) -> list[int]:
    """
    doc_list_str: a string of all doc id, title, content
    """
    client = get_gemini_client()
    prompt = dedent(f"""
        Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """).strip()
    response = client.models.generate_content(model=model, contents=prompt, config={
        "response_mime_type": "application/json"
    })
    response_text = response.text or ""
    # print("PROMPT",prompt)
    # print("response_text",response_text)
    ranked_ids = json.loads(response_text)
    ranked_ids = list(map(int, ranked_ids)) # make sure doc_id is int
    return ranked_ids


def rerank_batch(query:str,  search_result:list[tuple]) -> list[tuple[dict,float]]:
    doc_str_list = []
    res_map = dict()
    for res in search_result: # (score, doc, ...)
        doc = res[1]
        res_map[doc["id"]] = res
        doc_str_list.append(dedent(f""" 
            ID: {doc["id"]}
            Title: {doc["title"]}, 
            Description: {doc["description"]}
            """).strip()
        )
    doc_list_str = "\n\n--------------\n\n".join(doc_str_list)
    ranked_ids = llm_rerank_batch(query, doc_list_str)
    print('*'*20, "ranked_ids", ranked_ids)
    new_res = []
    for i, doc_id in enumerate(ranked_ids):
        new_res.append((res_map[doc_id], i+1))
    return new_res

def rerank_cross_encoder(query:str, search_result:list[tuple]) -> list[tuple[dict,float]]:
    pairs = []
    for res in search_result: # (score, doc, ...)
        doc = res[1]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    res_with_scores = sorted(list(zip(search_result, scores)), key=lambda x: x[1], reverse=True)
    new_res = []
    for res, score in res_with_scores:
        new_res.append((res, score))
    return new_res


def rerank(query:str, search_result:list[tuple], method:str, sleep_interval:int = 0) -> list[tuple[dict,float]]:
    match method:
        case "individual":
            return rerank_individual(query, search_result, sleep_interval)
        case "batch":
            return rerank_batch(query, search_result)
        case "cross_encoder":
            return rerank_cross_encoder(query, search_result)
        case _:
            raise ValueError("Unsupported")

