"""
Query engine for medical RAG system.
Based on blog post 2 code.
"""
from typing import Dict, List
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode

def create_query_engine(index_path, llm, embed_model, k=5, reranker=None):
    # Load the index
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # Configure the LLM
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Common kwargs
    qe_kwargs = {
        "response_mode": ResponseMode.TREE_SUMMARIZE,
        "text_qa_template": """
    You are a medical information assistant.
    Answer the question based ONLY on the following context.
    If you don't know the answer from the context, say "I don't have enough information to answer this question reliably. Please consult a healthcare professional."
    Do NOT make up or infer information not present in the context.
    Always cite the PMCID when providing information.

    Context:
    {context}

    Question: {query_str}

    Answer:""",
        "similarity_top_k": k,
    }

    # Add the reranker only if it's defined
    if reranker:
        qe_kwargs["node_postprocessors"] = [reranker]

    # Build the query engine
    query_engine = index.as_query_engine(**qe_kwargs)

    return query_engine


def query_medical_rag(question: str, query_engine, embed_model) -> str:
    """
    Query the medical RAG system (main function from blog post).
    
    Args:
        question: Question to ask
        query_engine: Configured query engine
    
    Returns:
        Formatted response with sources and disclaimer
    """
    Settings.embed_model = embed_model
    # Query the system
    response = query_engine.query(question)
    
    # Extract source documents
    source_nodes = response.source_nodes
    source_info = []
    
    for i, node in enumerate(source_nodes):
        source_info.append({
            "pmcid": node.metadata.get("pmcid", "Unknown"),
            "title": node.metadata.get("title", "Unknown"),
            "score": node.score if hasattr(node, "score") else None
        })
    
    # Format the final answer with metadata and disclaimer (from blog post)
    full_response = f"{response.response}\n\n"
    full_response += "Sources:\n"
    for i, source in enumerate(source_info):
        full_response += f"[{i+1}] PMCID {source['pmcid']} - {source['title']}\n"
    
    full_response += "\nReminder: This information is for educational purposes only and should not replace professional medical advice."
    
    return full_response