"""
RAG system for healthcare applications.
Simple modular version matching blog post 2.
"""

from .document_processor import DocumentProcessor
from .chunking import create_sentence_chunks, create_token_chunks, create_semantic_chunks
from .indexer import create_index
from .retriever import create_query_engine, query_medical_rag
from .evaluation import (
    create_pneumonia_test_questions,
    evaluate_rag_system, 
    run_full_evaluation,
    plot_evaluation_results
)
from .multi_stage import break_down_query, multi_stage_retrieval

__all__ = [
    "DocumentProcessor",
    "create_sentence_chunks",
    "create_token_chunks", 
    "create_semantic_chunks",
    "create_index",
    "create_query_engine",
    "query_medical_rag",
    "create_pneumonia_test_questions",
    "evaluate_rag_system",
    "run_full_evaluation", 
    "plot_evaluation_results",
    "break_down_query",
    "multi_stage_retrieval"
]