#Semantic entropy measurement for uncertainty detection.

import re
import math
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import debug_print


def calculate_semantic_entropy(question: str, query_engine, encoder, llm, num_samples: int = 5, 
                              temperature: float = 0.8) -> Dict:
    """
    Calculate semantic entropy to detect hallucination uncertainty.
    Higher entropy = more uncertainty = higher hallucination risk.
    
    Args:
        question: Question to ask
        query_engine: LlamaIndex query engine
        encoder: SentenceTransformer encoder
        llm: Language model (to adjust temperature)
        num_samples: Number of responses to generate
        temperature: Temperature for diversity
    
    Returns:
        Dictionary with entropy results
    """
    debug_print(f"=== CALCULATING SEMANTIC ENTROPY ===")
    debug_print(f"Generating {num_samples} responses with temperature={temperature}")
    
    # Generate multiple responses with higher temperature for diversity
    responses = []
    for i in range(num_samples):
        # Temporarily increase temperature for diversity
        original_temp = getattr(llm, 'temperature', 0.1)
        if hasattr(llm, 'temperature'):
            llm.temperature = temperature
        
        response = query_engine.query(question)
        responses.append(response.response)
        
        # Restore original temperature
        if hasattr(llm, 'temperature'):
            llm.temperature = original_temp
        
        debug_print(f"Response {i+1}: {response.response[:80]}...")
    
    # Sentence-level semantic clustering
    semantic_entropy = calculate_sentence_semantic_entropy(responses, encoder)
    
    # Interpretation
    if semantic_entropy >= 2.0:
        interpretation = "HIGH uncertainty - likely hallucination"
        confidence = "LOW"
    elif semantic_entropy >= 1.0:
        interpretation = "MEDIUM uncertainty - review recommended"
        confidence = "MEDIUM"
    else:
        interpretation = "LOW uncertainty - confident answer"
        confidence = "HIGH"
    
    debug_print(f"\nSemantic Entropy Score: {semantic_entropy:.3f}")
    debug_print(f"Interpretation: {interpretation}")
    
    return {
        'semantic_entropy': semantic_entropy,
        'responses': responses,
        'interpretation': interpretation,
        'high_uncertainty': semantic_entropy >= 1.5,
        'confidence': confidence
    }


def calculate_sentence_semantic_entropy(responses: List[str], encoder) -> float:
    """
    Calculate entropy based on semantic clustering of sentences.
    
    Args:
        responses: List of response strings
        encoder: SentenceTransformer encoder
    
    Returns:
        Semantic entropy score
    """
    # Extract all sentences from all responses
    all_sentences = []
    for response in responses:
        sentences = re.split(r'[.!?]', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) >= 10]
        all_sentences.extend(sentences)
    
    if len(all_sentences) < 2:
        return 0.0
    
    # Encode sentences
    embeddings = encoder.encode(all_sentences)
    
    # Simple clustering based on similarity threshold
    clusters = []
    used_indices = set()
    similarity_threshold = 0.7
    
    for i, emb_i in enumerate(embeddings):
        if i in used_indices:
            continue
        
        cluster = [i]
        used_indices.add(i)
        
        for j, emb_j in enumerate(embeddings):
            if j <= i or j in used_indices:
                continue
            
            similarity = cosine_similarity([emb_i], [emb_j])[0][0]
            if similarity > similarity_threshold:
                cluster.append(j)
                used_indices.add(j)
        
        clusters.append(cluster)
    
    # Calculate entropy based on cluster sizes
    cluster_sizes = [len(cluster) for cluster in clusters]
    total_sentences = len(all_sentences)
    
    # Calculate Shannon entropy
    entropy = 0.0
    for size in cluster_sizes:
        prob = size / total_sentences
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy