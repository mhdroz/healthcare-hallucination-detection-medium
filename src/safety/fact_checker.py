"""
External fact-checking module using Semantic Scholar.
Based on blog post 4 code.
"""
from typing import Dict, List, Tuple
from .external_sources import generate_scholar_keywords, search_semantic_scholar, prepare_abstract_sentences
from .attribution import check_answer_support
from app.utils import debug_print


def external_fact_check(answer: str, encoder, max_results: int = 10) -> Dict:
    """
    Fact-check an answer against external Semantic Scholar sources.
    From blog post 4.
    
    Args:
        answer: Answer text to fact-check
        encoder: SentenceTransformer encoder for similarity
        max_results: Maximum number of external sources to retrieve
    
    Returns:
        Dictionary with fact-checking results
    """
    debug_print("=== EXTERNAL FACT-CHECKING ===")
    
    try:
        # Step 1: Extract keywords from the answer
        debug_print("Extracting keywords for external search...")
        query = generate_scholar_keywords(answer)
        debug_print(f"Generated query: '{query}'")
        
        # Step 2: Search Semantic Scholar for abstracts
        debug_print("Searching Semantic Scholar for external sources...")
        abstracts = search_semantic_scholar(query, max_results=max_results)
        
        if not abstracts:
            return {
                "external_support_score": 0.0,
                "num_external_sources": 0,
                "query_used": query,
                "error": "No external sources found"
            }
        
        # Step 3: Prepare sentences from abstracts
        external_sentences = prepare_abstract_sentences(abstracts)
        
        if not external_sentences:
            return {
                "external_support_score": 0.0,
                "num_external_sources": len(abstracts),
                "query_used": query,
                "error": "No sentences extracted from abstracts"
            }
        
        # Step 4: Use existing attribution function to check support
        debug_print("Calculating similarity with external sources...")
        external_score, sentence_scores = check_answer_support(answer, external_sentences, encoder)
        
        debug_print(f"External fact-check score: {external_score:.3f}")
        
        return {
            "external_support_score": external_score,
            "sentence_scores": sentence_scores,
            "num_external_sources": len(abstracts),
            "num_external_sentences": len(external_sentences),
            "query_used": query,
            "abstracts_sample": abstracts[:3]  # Keep sample for reference
        }
        
    except Exception as e:
        debug_print(f"Error during external fact-checking: {e}")
        return {
            "external_support_score": 0.0,
            "num_external_sources": 0,
            "query_used": "",
            "error": str(e)
        }


def interpret_external_score(score: float) -> Dict:
    """
    Interpret external fact-checking score.
    
    Args:
        score: External support score (0-1)
    
    Returns:
        Interpretation dictionary
    """
    if score >= 0.7:
        interpretation = "Strong external validation - answer aligns well with recent literature"
        confidence = "HIGH"
    elif score >= 0.5:
        interpretation = "Moderate external validation - answer has some support in literature"
        confidence = "MEDIUM"  
    elif score >= 0.3:
        interpretation = "Weak external validation - limited support in recent literature"
        confidence = "LOW"
    else:
        interpretation = "Poor external validation - answer not well-supported by recent literature"
        confidence = "VERY LOW"
    
    return {
        "score": score,
        "interpretation": interpretation,
        "confidence": confidence
    }


def comprehensive_fact_check(answer: str, internal_sources: List[str], encoder, 
                           max_external_results: int = 10) -> Dict:
    """
    Comprehensive fact-checking using both internal and external sources.
    
    Args:
        answer: Answer to fact-check
        internal_sources: Internal source chunks from RAG
        encoder: SentenceTransformer encoder
        max_external_results: Max external sources to retrieve
    
    Returns:
        Comprehensive fact-checking results
    """
    debug_print("=== COMPREHENSIVE FACT-CHECKING ===")
    
    # Internal source attribution (from existing RAG sources)
    debug_print("Checking internal source attribution...")
    internal_score, internal_sentence_scores = check_answer_support(answer, internal_sources, encoder)
    
    # External fact-checking
    external_result = external_fact_check(answer, encoder, max_external_results)
    external_score = external_result.get("external_support_score", 0.0)
    
    # Combined analysis
    if external_result.get("error"):
        print(f" External fact-checking failed: {external_result['error']}")
        combined_score = internal_score  # Fall back to internal only
        reliability = "internal_only"
    else:
        # Weight internal sources more heavily (they're from our curated corpus)
        combined_score = (0.7 * internal_score) + (0.3 * external_score)
        reliability = "internal_and_external"
    
    # Interpret results
    internal_interp = interpret_external_score(internal_score)
    external_interp = interpret_external_score(external_score) if not external_result.get("error") else None
    
    debug_print(f"\nFACT-CHECKING SUMMARY:")
    debug_print(f"Internal Support: {internal_score:.3f} - {internal_interp['confidence']}")
    if external_interp:
        debug_print(f"External Support: {external_score:.3f} - {external_interp['confidence']}")
        debug_print(f"Combined Score: {combined_score:.3f}")
    else:
        debug_print(f"External Support: Failed")
        debug_print(f"Overall Score: {combined_score:.3f} (internal only)")
    
    return {
        "internal_score": internal_score,
        "internal_interpretation": internal_interp,
        "external_score": external_score,
        "external_interpretation": external_interp,
        "external_details": external_result,
        "combined_score": combined_score,
        "reliability": reliability,
        "recommendation": _get_fact_check_recommendation(combined_score, reliability)
    }


def _get_fact_check_recommendation(score: float, reliability: str) -> str:
    """Get recommendation based on fact-checking results."""
    if reliability == "internal_only":
        if score >= 0.7:
            return "Well-supported by internal sources, but external validation unavailable"
        elif score >= 0.5:
            return "Moderately supported by internal sources, consider seeking additional validation"
        else:
            return "Poorly supported by internal sources, high risk of inaccuracy"
    else:  # internal_and_external
        if score >= 0.7:
            return "Well-validated by both internal and external sources"
        elif score >= 0.5:
            return "Moderately validated, exercise caution in clinical application"
        else:
            return "Poorly validated, do not use without consulting healthcare professional"