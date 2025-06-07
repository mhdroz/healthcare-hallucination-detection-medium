#Index creation for RAG system.

from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings
from .chunking import create_sentence_chunks
from app.utils import debug_print


def create_index(documents: List[Document], embed_model, chunk_size: int = 512, 
                chunk_overlap: int = 50, index_path: str = None) -> VectorStoreIndex:
    """
    Create index from documents.
    
    Args:
        documents: List of Document objects
        embed_model: Embedding model to use
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks  
        index_path: Path to save index (optional)
    
    Returns:
        VectorStoreIndex
    """
    # Configure LlamaIndex settings
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    
    # Create nodes from documents with sentence splitter (default from blog)
    nodes = create_sentence_chunks(documents, chunk_size, chunk_overlap)
    
    # Create the vector index
    index = VectorStoreIndex(nodes)
    
    # Save the index to disk if path provided
    if index_path:
        index.storage_context.persist(index_path)
        debug_print(f"Created and saved index with {len(nodes)} nodes")
    
    return index
