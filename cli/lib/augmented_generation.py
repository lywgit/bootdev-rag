from textwrap import dedent
from .search_util import load_movie_list
from .gemini_util import (
    get_gemini_client,
    DEFAULT_GEMINI_MODEL
)
from .hybrid_search import HybridSearch


def llm_rag(query:str, docs:list, model=DEFAULT_GEMINI_MODEL) -> str:
    context = "\n".join([f" - {doc['title']}: {doc['description']} " for doc in docs])
    client = get_gemini_client()
    prompt = dedent(f"""
        Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {context}

        Provide a comprehensive answer that addresses the query:
        """).strip()

    response = client.models.generate_content(model=model, contents=prompt)
    response_text = response.text or ""
    return response_text
    
def llm_summarize(query:str, docs:list, model=DEFAULT_GEMINI_MODEL) -> str:
    context = "\n".join([f" - {doc['title']}: {doc['description']} " for doc in docs])
    client = get_gemini_client()
    prompt = dedent(f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {context}
        Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
        """).strip()

    response = client.models.generate_content(model=model, contents=prompt)
    response_text = response.text or ""
    return response_text

def llm_citations(query:str, docs:list, model=DEFAULT_GEMINI_MODEL) -> str:
    context = "\n".join([f" - {i+1}. {doc['title']}: {doc['description']} " for i, doc in enumerate(docs)])
    client = get_gemini_client()
    prompt = dedent(f"""
        Answer the question or provide information based on the provided documents.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {context}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:
        """).strip()

    response = client.models.generate_content(model=model, contents=prompt)
    response_text = response.text or ""
    return response_text


def llm_question(question:str, docs:list, model=DEFAULT_GEMINI_MODEL) -> str:
    context = "\n".join([f" - {i+1}. {doc['title']}: {doc['description']} " for i, doc in enumerate(docs)])
    client = get_gemini_client()
    prompt = dedent(f"""
        Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {question}

        Documents:
        {context}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:
        """).strip()

    response = client.models.generate_content(model=model, contents=prompt)
    response_text = response.text or ""
    return response_text


def rag_command(query:str) -> dict:
    docs = load_movie_list()
    hs = HybridSearch(docs)
    search_res = hs.rrf_search(query, limit=5)
    retrieved_docs = [res[1] for res in search_res]
    rag_response = llm_rag(query, retrieved_docs)
    return {
        'search_results':retrieved_docs,
        'rag_answer': rag_response
        }


def summarize_command(query:str, limit:int) -> dict:
    docs = load_movie_list()
    hs = HybridSearch(docs)
    search_res = hs.rrf_search(query, limit=limit)
    retrieved_docs = [res[1] for res in search_res]
    rag_response = llm_summarize(query, retrieved_docs)
    return {
        'search_results':retrieved_docs,
        'rag_summary': rag_response
        }

def citations_command(query:str, limit:int) -> dict:
    docs = load_movie_list()
    hs = HybridSearch(docs)
    search_res = hs.rrf_search(query, limit=limit)
    retrieved_docs = [res[1] for res in search_res]
    rag_response = llm_citations(query, retrieved_docs)
    return {
        'search_results':retrieved_docs,
        'rag_citations': rag_response
        }

def question_command(question:str, limit:int) -> dict:
    docs = load_movie_list()
    hs = HybridSearch(docs)
    search_res = hs.rrf_search(question, limit=limit)
    retrieved_docs = [res[1] for res in search_res]
    rag_response = llm_question(question, retrieved_docs)
    return {
        'search_results':retrieved_docs,
        'rag_answer': rag_response
        }