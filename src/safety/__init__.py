"""
Safety checks for healthcare RAG systems.
Based on blog post 3 - hallucination detection methods + blog post 4 external fact-checking.
"""

from .attribution import check_answer_support, find_weak_sentences
from .consistency import check_consistency
from .entropy import calculate_semantic_entropy
from .multi_stage import break_down_query, multi_stage_retrieval
from .external_sources import generate_scholar_keywords, search_semantic_scholar, prepare_abstract_sentences
from .fact_checker import external_fact_check, comprehensive_fact_check
from .safety_checker import comprehensive_safety_check

__all__ = [
    "check_answer_support",
    "find_weak_sentences", 
    "check_consistency",
    "calculate_semantic_entropy",
    "break_down_query",
    "multi_stage_retrieval",
    "generate_scholar_keywords",
    "search_semantic_scholar", 
    "prepare_abstract_sentences",
    "external_fact_check",
    "comprehensive_fact_check",
    "comprehensive_safety_check"
]