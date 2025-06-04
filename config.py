"""
Configuration settings for the Healthcare RAG system.
"""
import os
from pathlib import Path


class Config:
    """Configuration class for the Healthcare RAG system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INDICES_DIR = DATA_DIR / "indices"
    
    # PubMed/NCBI settings
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")
    EMAIL = os.getenv("EMAIL")
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
    
    # Embedding models
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    BIOMEDICAL_EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    
    # LLM settings
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.1
    HIGH_TEMPERATURE = 0.8  # For semantic entropy
    
    # Safety thresholds
    ATTRIBUTION_THRESHOLD = 0.6
    CONSISTENCY_THRESHOLD = 0.6
    SEMANTIC_ENTROPY_HIGH = 2.0
    SEMANTIC_ENTROPY_MEDIUM = 1.0
    
    # File paths
    DEFAULT_CORPUS_FILE = "pmc_articles.jsonl"
    DEFAULT_INDEX_NAME = "medical_index"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INDICES_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_corpus_path(cls, filename: str = None) -> Path:
        """Get the full path to a corpus file."""
        filename = filename or cls.DEFAULT_CORPUS_FILE
        return cls.PROCESSED_DATA_DIR / filename
    
    @classmethod
    def get_index_path(cls, index_name: str = None) -> Path:
        """Get the full path to an index directory."""
        index_name = index_name or cls.DEFAULT_INDEX_NAME
        return cls.INDICES_DIR / index_name