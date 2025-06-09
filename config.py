"""
Configuration settings for the Healthcare RAG system.
"""
import os
from pathlib import Path

DEBUG_MODE = True

# Embedding models
# Make sure you choose the same embedding model you used to build your index
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  

# LLMs
OPENAI_MODEL_NAME = "gpt-4o-2024-11-20"
#OPENAI_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
HIGH_TEMPERATURE = 0.8 

# Corpus 
CORPUS_PATH = "data/processed/expanded_pneumonia.jsonl"
INDEX_PATH = "./data/indices/pneumonia_index"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# RAG settings
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
DEFAULT_SIMILARITY_TOP_K = 5
EXTENDED_SIMILARITY_TOP_K = 12
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"

# Semantic Entropy samples
NUM_SAMPLES_ENTROPY = 3

# External fact checking
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"