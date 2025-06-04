"""
Chunking strategies for medical documents.
Based on blog post 2 code - simple chunking options.
"""
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode


def create_sentence_chunks(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[BaseNode]:
    """Create chunks using sentence splitter (default from blog post)."""
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes using sentence splitting")
    return nodes


def create_token_chunks(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[BaseNode]:
    """Create chunks using token splitter (alternative mentioned in blog post)."""
    token_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = token_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes using token splitting")
    return nodes


def create_semantic_chunks(documents: List[Document], embed_model, buffer_size: int = 1, 
                          breakpoint_percentile_threshold: int = 95) -> List[BaseNode]:
    """Create chunks using semantic splitter (advanced option from blog post)."""
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model
    )
    nodes = semantic_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes using semantic splitting")
    return nodes