"""
Multi-stage retrieval for complex medical queries.
Based on blog post 3 code.
"""
import re
from typing import List, Dict
from app.utils import debug_print


def break_down_query(complex_question: str, llm) -> List[str]:
    """
    Break a complex medical question into simpler parts.
    From blog post 3.
    
    Args:
        complex_question: Complex medical question
        llm: Language model for query decomposition
    
    Returns:
        List of simpler sub-questions
    """
    prompt = f"""You are a medical librarian. Break down this complex medical question into 2-4 simpler, specific questions that together would provide a complete answer.

Complex question: {complex_question}

Provide the simpler questions as a numbered list:
1.
2.
3.
4.
"""
    
    response = llm.complete(prompt)
    debug_print("=== QUERY BREAKDOWN ===")
    debug_print(response.text)
    
    # Extract the sub-questions
    lines = response.text.strip().split('\n')
    sub_questions = []
    
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering
            clean_question = re.sub(r'^[\d\-\.\)\s]+', '', line).strip()
            if clean_question:
                sub_questions.append(clean_question)
    
    return sub_questions


def multi_stage_retrieval(complex_question: str, query_engine, llm) -> Dict:
    """
    Perform multi-stage retrieval for complex questions.
    From blog post 3.
    
    Args:
        complex_question: Complex question requiring multiple sources
        query_engine: LlamaIndex query engine
        llm: Language model for decomposition and synthesis
    
    Returns:
        Dictionary with multi-stage results
    """
    debug_print("=== MULTI-STAGE RETRIEVAL ===")
    debug_print("=" * 50)
    
    # Step 1: Break down the question
    sub_questions = break_down_query(complex_question, llm)
    
    # Step 2: Get answers for each sub-question
    sub_answers = []
    all_sources = []
    
    for i, sub_q in enumerate(sub_questions):
        debug_print(f"\n--- Sub-question {i+1}: {sub_q} ---")
        response = query_engine.query(sub_q)
        sub_answer = response.response
        sources = response.source_nodes
        
        debug_print(f"Answer: {sub_answer[:150]}...")
        
        sub_answers.append({
            'question': sub_q,
            'answer': sub_answer,
            'sources': sources
        })
        
        # Collect unique sources
        for source in sources:
            if source.text not in [s.text for s in all_sources]:
                all_sources.append(source)
    
    # Step 3: Synthesize final answer
    debug_print(f"\n=== SYNTHESIZING FINAL ANSWER ===")
    context = ""
    for i, sub in enumerate(sub_answers):
        context += f"Sub-question {i+1}: {sub['question']}\n"
        context += f"Answer: {sub['answer']}\n\n"
    
    synthesis_prompt = f"""Based on the following information, provide a comprehensive answer to the original question.

Original question: {complex_question}

Information gathered:
{context}

Instructions:
• Combine the information into one coherent answer
• Only use the information provided above
• If there are contradictions, mention them
• Be specific and cite relevant details

Comprehensive answer:
"""
    
    final_response = llm.complete(synthesis_prompt)
    final_answer = final_response.text
    
    debug_print("Final synthesized answer:")
    debug_print(final_answer)
    
    return {
        'original_question': complex_question,
        'sub_questions': sub_questions,
        'sub_answers': sub_answers,
        'final_answer': final_answer,
        'all_sources': all_sources
    }