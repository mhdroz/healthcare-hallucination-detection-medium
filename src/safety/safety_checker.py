"""
Comprehensive safety checker combining all safety methods.
Based on blog post 3 code + external fact-checking from blog post 4.
"""
from typing import Dict
from .attribution import check_answer_support, find_weak_sentences
from .consistency import check_consistency
from .entropy import calculate_semantic_entropy
from .multi_stage import multi_stage_retrieval
from .fact_checker import comprehensive_fact_check
import config as cfg
from app.utils import debug_print


def comprehensive_safety_check(question: str, query_engine, llm, encoder, num_tries: int = 3, 
                               use_multi_stage: bool = False, enable_fact_check: bool = True) -> Dict:
    """
    Perform comprehensive safety checking on a RAG response.
    Enhanced version from blog post 3 + external fact-checking from blog post 4.
    
    Args:
        question: Question to check
        query_engine: LlamaIndex query engine
        llm: Language model
        encoder: SentenceTransformer encoder
        num_tries: Number of consistency checks
        use_multi_stage: Whether to use multi-stage retrieval
        enable_fact_check: Whether to run external fact-checking
    
    Returns:
        Comprehensive safety assessment
    """
    debug_print("=== COMPREHENSIVE MEDICAL RAG SAFETY CHECK ===")
    debug_print("=" * 60)
    
    # Step 1: Get the answer
    if use_multi_stage:
        debug_print("Using multi-stage retrieval...")
        result = multi_stage_retrieval(question, query_engine, llm)
        answer = result["final_answer"]
        source_nodes = result.get("all_sources", [])
    else:
        debug_print("Using standard retrieval...")
        response = query_engine.query(question)
        answer = response.response
        source_nodes = response.source_nodes

    # Format source chunks into a serializable list of dicts
    source_chunks = [
        {
            "text": node.text,
            "score": node.score,
            "pmcid": node.metadata.get("pmcid", "N/A"),
            "title": node.metadata.get("title", "No Title")
        }
        for node in source_nodes
    ]
    
    debug_print(f"\nQuestion: {question}")
    debug_print(f"Answer: {answer[:200]}...")
    
    # Step 2: Attribution check
    debug_print(f"\n=== ATTRIBUTION CHECK ===")
    attribution_score, _ = check_answer_support(answer, [chunk['text'] for chunk in source_chunks], encoder)
    
    # Step 3: Consistency check
    debug_print(f"\n=== CONSISTENCY CHECK ===")
    consistency_score, _ = check_consistency(question, query_engine, encoder, num_tries=num_tries)
    
    # Step 4: Find weak sentences
    debug_print(f"\n=== WEAK SENTENCE DETECTION ===")
    weak_sentences = find_weak_sentences(answer, [chunk['text'] for chunk in source_chunks], encoder)
    
    # Show warnings
    if weak_sentences:
        debug_print(f"\nWEAK SENTENCES DETECTED")
        for weak in weak_sentences:
            debug_print(f"• Weak support: \"{weak['sentence'][:100]}...\"")
    else:
        debug_print("\nNo weak sentences detected\n")
    
    # Step 5: Calculate semantic entropy
    entropy_result = calculate_semantic_entropy(
        question, query_engine, encoder, llm, num_samples=3, temperature=cfg.HIGH_TEMPERATURE
    )
    semantic_entropy = entropy_result['semantic_entropy']
    
    # Step 6: External fact-checking (NEW from blog post 4)
    fact_check_result = None
    if enable_fact_check:
        debug_print(f"\n=== EXTERNAL FACT-CHECKING ===")
        try:
            fact_check_result = comprehensive_fact_check(answer, [chunk['text'] for chunk in source_chunks], encoder)
        except Exception as e:
            debug_print(f"External fact-checking failed: {e}")
            fact_check_result = {"error": str(e)}
    
    # Step 7: Overall safety assessment (enhanced)
    debug_print(f"\n=== OVERALL SAFETY ASSESSMENT ===")
    debug_print("=" * 40)
    
    safety_score = 0
    max_score = 4 if enable_fact_check and fact_check_result and not fact_check_result.get("error") else 3
    
    debug_print(f"Attribution Score: {attribution_score:.3f}")
    if attribution_score >= 0.6:
        safety_score += 1
        debug_print("Good source attribution")
    else:
        debug_print("Weak source attribution")
    
    debug_print(f"Consistency Score: {consistency_score:.3f}")
    if consistency_score >= 0.6:
        safety_score += 1
        debug_print("Good consistency")
    else:
        debug_print("Low consistency")
    
    debug_print(f"Semantic entropy score: {semantic_entropy:.3f}")
    if entropy_result['confidence'] == "HIGH":
        safety_score += 1
        debug_print("Good semantic entropy")
    else:
        debug_print("High semantic entropy")
    
    # External fact-checking score
    if enable_fact_check and fact_check_result and not fact_check_result.get("error"):
        external_score = fact_check_result.get("combined_score", 0.0)
        debug_print(f"External Validation: {external_score:.3f}")
        if external_score >= 0.6:
            safety_score += 1
            debug_print("Good external validation")
        else:
            debug_print("Weak external validation")
    elif enable_fact_check:
        debug_print("External validation failed")
    
    # Final confidence level
    confidence_ratio = safety_score / max_score
    if confidence_ratio >= 0.75:
        confidence = "HIGH CONFIDENCE"
    elif confidence_ratio >= 0.5:
        confidence = "MEDIUM CONFIDENCE"
    else:
        confidence = "LOW CONFIDENCE"
    
    debug_print(f"\nFinal Assessment: {confidence} ({safety_score}/{max_score})")
    debug_print(f"\nMedical Disclaimer: This information is for educational purposes only.")
    
    return {
        "question": question,
        "answer": answer,
        "attribution_score": attribution_score,
        "consistency_score": consistency_score,
        "semantic_entropy": semantic_entropy,
        "weak_sentences": weak_sentences,
        "fact_check_result": fact_check_result,
        "safety_score": safety_score,
        "max_safety_score": max_score,
        "confidence": confidence,
        "use_multi_stage": use_multi_stage,
        "external_fact_check_enabled": enable_fact_check,
        "source_chunks": source_chunks
    }