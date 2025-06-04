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
    print("=== COMPREHENSIVE MEDICAL RAG SAFETY CHECK ===")
    print("=" * 60)
    
    # Step 1: Get the answer
    if use_multi_stage:
        print("Using multi-stage retrieval...")
        result = multi_stage_retrieval(question, query_engine, llm)
        answer = result["final_answer"]
        source_chunks = [node.text for node in result["all_sources"]]
    else:
        print("Using standard retrieval...")
        response = query_engine.query(question)
        answer = response.response
        source_chunks = [node.text for node in response.source_nodes]
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer[:200]}...")
    
    # Step 2: Attribution check
    print(f"\n=== ATTRIBUTION CHECK ===")
    attribution_score, _ = check_answer_support(answer, source_chunks, encoder)
    
    # Step 3: Consistency check
    print(f"\n=== CONSISTENCY CHECK ===")
    consistency_score, _ = check_consistency(question, query_engine, encoder, num_tries=num_tries)
    
    # Step 4: Find weak sentences
    print(f"\n=== WEAK SENTENCE DETECTION ===")
    weak_sentences = find_weak_sentences(answer, source_chunks, encoder)
    
    # Show warnings
    if weak_sentences:
        print(f"\n⚠️  WEAK SENTENCES DETECTED")
        for weak in weak_sentences:
            print(f"• Weak support: \"{weak['sentence'][:100]}...\"")
    else:
        print("\nNo weak sentences detected\n")
    
    # Step 5: Calculate semantic entropy
    entropy_result = calculate_semantic_entropy(
        question, query_engine, encoder, llm, num_samples=3
    )
    semantic_entropy = entropy_result['semantic_entropy']
    
    # Step 6: External fact-checking (NEW from blog post 4)
    fact_check_result = None
    if enable_fact_check:
        print(f"\n=== EXTERNAL FACT-CHECKING ===")
        try:
            fact_check_result = comprehensive_fact_check(answer, source_chunks, encoder)
        except Exception as e:
            print(f"⚠️  External fact-checking failed: {e}")
            fact_check_result = {"error": str(e)}
    
    # Step 7: Overall safety assessment (enhanced)
    print(f"\n=== OVERALL SAFETY ASSESSMENT ===")
    print("=" * 40)
    
    safety_score = 0
    max_score = 4 if enable_fact_check and fact_check_result and not fact_check_result.get("error") else 3
    
    print(f"Attribution Score: {attribution_score:.3f}")
    if attribution_score >= 0.6:
        safety_score += 1
        print("✅ Good source attribution")
    else:
        print("❌ Weak source attribution")
    
    print(f"Consistency Score: {consistency_score:.3f}")
    if consistency_score >= 0.6:
        safety_score += 1
        print("✅ Good consistency")
    else:
        print("❌ Low consistency")
    
    print(f"Semantic entropy score: {semantic_entropy:.3f}")
    if entropy_result['confidence'] == "HIGH":
        safety_score += 1
        print("✅ Good semantic entropy")
    else:
        print("❌ High semantic entropy")
    
    # External fact-checking score
    if enable_fact_check and fact_check_result and not fact_check_result.get("error"):
        external_score = fact_check_result.get("combined_score", 0.0)
        print(f"External Validation: {external_score:.3f}")
        if external_score >= 0.6:
            safety_score += 1
            print("✅ Good external validation")
        else:
            print("❌ Weak external validation")
    elif enable_fact_check:
        print("❌ External validation failed")
    
    # Final confidence level
    confidence_ratio = safety_score / max_score
    if confidence_ratio >= 0.75:
        confidence = "HIGH CONFIDENCE"
    elif confidence_ratio >= 0.5:
        confidence = "MEDIUM CONFIDENCE"
    else:
        confidence = "LOW CONFIDENCE"
    
    print(f"\nFinal Assessment: {confidence} ({safety_score}/{max_score})")
    print(f"\nMedical Disclaimer: This information is for educational purposes only.")
    
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
        "external_fact_check_enabled": enable_fact_check
    }