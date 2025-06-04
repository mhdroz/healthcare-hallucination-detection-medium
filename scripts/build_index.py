#!/usr/bin/env python3
"""
Simple CLI script to build RAG index.
Based on blog post 2 code.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag import DocumentProcessor, create_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from config import Config
import os
from dotenv import load_dotenv


def main():
    """Build RAG index from corpus."""
    parser = argparse.ArgumentParser(description="Build RAG index from medical corpus")
    
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--index-path", type=str, default="./medical_index", help="Path to save index")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    print(f"Loading corpus: {args.corpus}")
    print(f"Index path: {args.index_path}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    
    # Load and process documents (from blog post)
    articles = DocumentProcessor.load_medical_articles(args.corpus)
    documents = DocumentProcessor.process_articles(articles)
    print(f"Created {len(documents)} documents from {len(articles)} articles")
    
    # Use embedding model from blog post
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Create index (main function from blog post)
    index = create_index(
        documents=documents,
        embed_model=embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_path=args.index_path
    )
    
    print(f"✅ Index built and saved to: {args.index_path}")
    
    # Quick test (from blog post)
    from rag import create_query_engine, query_medical_rag
    
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    query_engine = create_query_engine(args.index_path, llm)
    
    question = "What are the common treatments for bacterial pneumonia?"
    answer = query_medical_rag(question, query_engine)
    
    print("\n" + "="*50)
    print("QUICK TEST")
    print("="*50)
    print(f"Question: {question}")
    print(f"Answer: {answer[:300]}...")


if __name__ == "__main__":
    main()