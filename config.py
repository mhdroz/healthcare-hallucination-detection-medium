"""
Configuration settings for the Healthcare RAG system.
"""
import os
from pathlib import Path

DEBUG_MODE = True

 # Embedding models
#DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Biomedical
EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  

# LLMs
OPENAI_MODEL_NAME = "gpt-4o-2024-11-20"
#OPENAI_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
HIGH_TEMPERATURE = 0.8 

# Corpus 
CORPUS_PATH = "pmc_articles.jsonl"
INDEX_PATH = "./data/indices/pneumonia_index"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Default corpus settings
DEFAULT_QUERY = "infectious disease OR pneumonia OR sepsis"
DEFAULT_MAX_RESULTS = 500
DEFAULT_BATCH_SIZE = 50
DEFAULT_DELAY = 0.2
DEFAULT_ALLOWED_LICENSES = {"cc-by", "cc-by-sa", "cc0"}

# RAG settings
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SIMILARITY_TOP_K = 5
USE_RERANKER = False
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"

# Safety thresholds
ATTRIBUTION_THRESHOLD = 0.6
CONSISTENCY_THRESHOLD = 0.6
SEMANTIC_ENTROPY_HIGH = 2.0
SEMANTIC_ENTROPY_MEDIUM = 1.0

# External fact checking
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"