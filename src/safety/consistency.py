"""
Consistency checking for RAG responses.
Based on blog post 3 code.
"""
import time
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import debug_print


def check_consistency(question: str, query_engine, encoder, num_tries: int = 3) -> Tuple[float, List[str]]:
    """
    Ask the same question multiple times and check for consistency.
    From blog post 3.
    
    Args:
        question: Question to ask repeatedly
        query_engine: LlamaIndex query engine
        encoder: SentenceTransformer encoder for embeddings
        num_tries: Number of times to ask the question
    
    Returns:
        Tuple of (consistency_score, all_responses)
    """
    debug_print(f"Asking the same question {num_tries} times...")
    
    responses = []
    for i in range(num_tries):
        debug_print(f"Attempt {i+1}...")
        response = query_engine.query(question)
        responses.append(response.response)
        time.sleep(1)  # Brief pause between queries
    
    # Show all responses
    debug_print("\n=== ALL RESPONSES ===")
    for i, resp in enumerate(responses):
        debug_print(f"Response {i+1}: {resp[:100]}...")
        debug_print()
    
    # Calculate similarity between responses
    if len(responses) < 2:
        return 1.0, responses
    
    response_embeddings = encoder.encode(responses)
    similarities = []
    
    for i in range(len(response_embeddings)):
        for j in range(i + 1, len(response_embeddings)):
            sim = cosine_similarity([response_embeddings[i]], [response_embeddings[j]])[0][0]
            similarities.append(sim)
            debug_print(f"Similarity between response {i+1} and {j+1}: {sim:.3f}")
    
    avg_similarity = np.mean(similarities)
    debug_print(f"\nAverage consistency score: {avg_similarity:.3f}")
    
    if avg_similarity >= 0.8:
        debug_print("High consistency - responses are very similar")
    elif avg_similarity >= 0.6:
        debug_print("Moderate consistency - some variation")
    else:
        debug_print("Low consistency - significant differences (potential hallucination risk)")
    
    return avg_similarity, responses