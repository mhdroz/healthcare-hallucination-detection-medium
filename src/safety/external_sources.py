"""
External source integration for fact-checking.
Based on blog post 4 code.
"""
import os
import re
import requests
from typing import List
from openai import OpenAI


def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    Call OpenAI API with system and user prompts.
    From blog post 4.
    
    Args:
        system_prompt: System instruction
        user_prompt: User message
        model: OpenAI model to use
        temperature: Generation temperature
    
    Returns:
        Generated response text
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_scholar_keywords(answer: str) -> str:
    """
    Generate keywords for Semantic Scholar search from an answer.
    From blog post 4.
    
    Args:
        answer: Answer text to extract keywords from
    
    Returns:
        Keywords string for search
    """
    system_prompt = """You are a clinical NLP assistant that extracts 3–6 concise keywords
from an answer so they can be used as a Semantic Scholar search query.

Rules
1. Output ONE line containing only the keywords separated by spaces.
2. Use lower-case nouns; drop adjectives and stop-words.
3. Include a keyword for:
   • the main disease / problem
   • the intervention / drug class (if present)
   • the population or special setting (if present)
4. Do NOT include numbers, punctuation, extra words, or explanations.
5. If the answer covers multiple distinct topics, pick the MOST
   central one (usually the first sentence).
"""

    user_prompt = f"""Extract keywords for Semantic Scholar from this answer:

{answer}

Keywords:
"""
    
    keywords = call_openai(system_prompt, user_prompt)
    return keywords.strip()


def search_semantic_scholar(query: str, max_results: int = 10) -> List[str]:
    """
    Search Semantic Scholar for abstracts.
    From blog post 4.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
    
    Returns:
        List of abstract texts
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": max_results, "fields": "title,abstract"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json()

        abstracts = []
        for article in articles.get('data', []):
            if article.get('abstract'):
                abstracts.append(article['abstract'])
            else:
                print("No abstract available for one article")
        
        print(f"Retrieved {len(abstracts)} abstracts from Semantic Scholar")
        return abstracts
        
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []


def _split_into_sentences(text: str, min_len: int = 10) -> List[str]:
    """
    Lightweight sentence splitter.
    From blog post 4.
    
    Args:
        text: Text to split into sentences
        min_len: Minimum sentence length
    
    Returns:
        List of sentences
    """
    sentence_split_re = re.compile(r'(?<=[.!?])\s+')
    pieces = sentence_split_re.split(text)
    return [s.strip() for s in pieces if len(s.strip()) >= min_len]


def prepare_abstract_sentences(abstracts: List[str], min_len: int = 10) -> List[str]:
    """
    Convert abstracts to sentences for similarity scoring.
    From blog post 4.
    
    Args:
        abstracts: List of abstract texts
        min_len: Minimum sentence length to keep
    
    Returns:
        Flattened list of sentences
    """
    sentences = []
    for abs_text in abstracts:
        sentences.extend(_split_into_sentences(abs_text, min_len=min_len))
    
    print(f"Prepared {len(sentences)} sentences from {len(abstracts)} abstracts")
    return sentences