#FastAPI web application for transparent healthcare RAG system.
#Building trustworthy healthcare LLM systems - Part 4.

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import our modules

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, StorageContext, load_index_from_storage



from app.routes import router

import config as cfg

# Load environment
load_dotenv()

async def initialize_models(app: FastAPI):
    """Initialize all models on startup and store them in app.state."""
    print("🚀 Initializing healthcare RAG system...")
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY required")
    
    # Initialize models
    embed_model = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)
    Settings.embed_model = embed_model # Set the global settings

    app.state.llm = OpenAI(model=cfg.OPENAI_MODEL_NAME, temperature=cfg.DEFAULT_TEMPERATURE)
    app.state.encoder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    app.state.embed_model = embed_model
    
    # Load index (configurable path)
    index_path = cfg.INDEX_PATH
    if not os.path.exists(index_path):
        raise ValueError(f"Index not found at {index_path}. Please build an index first.")
    
        # Load index
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    
    app.state.index = index

    print("✅ Models initialized successfully!")

    
async def cleanup_models():
    """Cleanup models on shutdown."""
    print("🧹 Cleaning up models...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await initialize_models(app)
    yield
    # Shutdown
    await cleanup_models()


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Healthcare RAG System",
    description="Transparent Medical AI with Safety Checks",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.include_router(router)
