"""
Source attribution scoring for RAG responses.
Based on blog post 3 code.
"""
import re
import numpy as np
from typing import List, Tuple
from app.utils import debug_print
from sklearn.metrics.pairwise import cosine_similarity


def check_answer_support(answer: str, source_chunks: List[str], encoder) -> Tuple[float, List[float]]:
    """
    Simple function to check how well an answer is supported by source chunks.
    From blog post 3.
    
    Args:
        answer: The generated answer text
        source_chunks: List of retrieved source text chunks
        encoder: SentenceTransformer encoder for embeddings
    
    Returns:
        Tuple of (overall_score, sentence_scores)
    """
    # Split answer into sentences (rough approximation as noted in blog)
    sentences = re.split(r'[.!?]', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences or not source_chunks:
        return 0.0, []
    
    print(f"Checking {len(sentences)} sentences against {len(source_chunks)} source chunks")
    
    # Encode sentences and sources
    answer_embeddings = encoder.encode(sentences)
    source_embeddings = encoder.encode(source_chunks)
    
    sentence_scores = []
    for i, answer_emb in enumerate(answer_embeddings):
        # Find best matching source for each sentence
        similarities = cosine_similarity([answer_emb], source_embeddings)[0]
        best_score = np.max(similarities)
        sentence_scores.append(best_score)
        debug_print(f"Sentence {i+1}: '{sentences[i][:50]}...' → Score: {best_score:.3f}")
    
    overall_score = np.mean(sentence_scores)
    return overall_score, sentence_scores


def find_weak_sentences(answer: str, source_chunks: List[str], encoder, threshold: float = 0.5) -> List[dict]:
    """
    Identify sentences that might be hallucinated (poorly supported by sources).
    From blog post 3.
    
    Args:
        answer: The generated answer text
        source_chunks: List of retrieved source text chunks  
        encoder: SentenceTransformer encoder
        threshold: Minimum similarity threshold
    
    Returns:
        List of weak sentences with scores
    """
    sentences = re.split(r'[.!?]', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences or not source_chunks:
        return []
    
    # Get similarity scores
    answer_embeddings = encoder.encode(sentences)
    source_embeddings = encoder.encode(source_chunks)
    
    weak_sentences = []
    for i, answer_emb in enumerate(answer_embeddings):
        similarities = cosine_similarity([answer_emb], source_embeddings)[0]
        best_score = np.max(similarities)
        
        if best_score < threshold:
            weak_sentences.append({
                'sentence': sentences[i],
                'score': best_score,
                'index': i
            })
    
    return weak_sentences