"""
FastAPI web application for transparent healthcare RAG system.
Building trustworthy healthcare LLM systems - Part 4.
"""
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import our modules
from rag import create_query_engine, run_full_evaluation
from safety import comprehensive_safety_check
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

import config as cfg

# Load environment
load_dotenv()

# Global variables for models
query_engine = None
llm = None
encoder = None
embed_model = HuggingFaceEmbedding(model_name=cfg.DEFAULT_EMBEDDING_MODEL)

Settings.embed_model = embed_model


async def initialize_models():
    """Initialize all models on startup."""
    global query_engine, llm, encoder, embed_model
    
    print("🚀 Initializing healthcare RAG system...")
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY required")
    
    # Initialize models
    embed_model = HuggingFaceEmbedding(model_name=cfg.DEFAULT_EMBEDDING_MODEL)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    encoder = SentenceTransformer(cfg.DEFAULT_EMBEDDING_MODEL)
    
    # Load index (configurable path)
    index_path = os.getenv("INDEX_PATH", "./data/indices/medical_index")
    if not os.path.exists(index_path):
        raise ValueError(f"Index not found at {index_path}. Please build an index first.")
    
    query_engine = create_query_engine(index_path, llm, embed_model)
    
    print("✅ Models initialized successfully!")


async def cleanup_models():
    """Cleanup models on shutdown."""
    print("🧹 Cleaning up models...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await initialize_models()
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

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Medical question to analyze")
    multi_stage: bool = Field(False, description="Use multi-stage retrieval")
    fact_check: bool = Field(True, description="Enable external fact-checking")
    consistency_tries: int = Field(3, ge=2, le=10, description="Number of consistency checks")


class SafetyResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    safety_score: int
    max_safety_score: int
    attribution_score: float
    consistency_score: float
    semantic_entropy: float
    weak_sentences: list
    has_weak_sentences: bool
    multi_stage_used: bool
    fact_check_enabled: bool
    external_validation: Optional[dict] = None
    safety_interpretations: dict


class EvaluationResponse(BaseModel):
    faithfulness_score: float
    relevancy_score: float
    faithfulness_interpretation: str
    relevancy_interpretation: str
    overall_grade: str
    num_questions: int
    detailed_scores: list


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main interface page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_model=SafetyResponse)
async def handle_query(query_request: QueryRequest):
    """Handle medical queries with full safety analysis."""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Models not initialized")
        
        print(f"🔍 Processing query: {query_request.question}")
        
        # Run comprehensive safety check
        safety_result = comprehensive_safety_check(
            question=query_request.question,
            query_engine=query_engine,
            llm=llm,
            encoder=encoder,
            num_tries=query_request.consistency_tries,
            use_multi_stage=query_request.multi_stage,
            enable_fact_check=query_request.fact_check
        )
        
        # Format response
        response = format_safety_response(safety_result)
        return SafetyResponse(**response)
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def handle_evaluation():
    """Run RAGAS evaluation on the system."""
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        print("📊 Running RAGAS evaluation...")
        
        judge_llm = OpenAI(model="gpt-4o", temperature=0.1)
        
        # Run evaluation with default pneumonia questions
        results = run_full_evaluation(
            query_engine=query_engine,
            judge_llm=judge_llm,
            show_plot=False  # Don't show plot in web context
        )
        
        # Format evaluation results
        summary = results["summary"]
        interpretation = results["interpretation"]
        
        eval_response = {
            "faithfulness_score": summary["faithfulness_mean"],
            "relevancy_score": summary["answer_relevancy_mean"],
            "faithfulness_interpretation": interpretation["faithfulness_interpretation"],
            "relevancy_interpretation": interpretation["relevancy_interpretation"],
            "overall_grade": interpretation["overall_grade"],
            "num_questions": summary["num_questions"],
            "detailed_scores": summary["detailed_scores"]
        }
        
        return EvaluationResponse(**eval_response)
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=query_engine is not None
    )


@app.get("/api/sample-questions")
async def get_sample_questions():
    """Get sample questions for testing."""
    samples = [
        "What are the common treatments for bacterial pneumonia?",
        "When should azithromycin dose be adjusted in moderate renal impairment?",
        "What antibiotics are safe to use with warfarin in elderly patients?",
        "What is the first-line antibiotic regimen for community-acquired pneumonia?",
        "How does recommended empiric therapy change for pneumonia in regions with resistant S. pneumoniae?"
    ]
    return {"samples": samples}


def format_safety_response(safety_result):
    """Format safety check results for the frontend."""
    # Basic response data
    response = {
        'question': safety_result['question'],
        'answer': safety_result['answer'],
        'confidence': safety_result['confidence'],
        'safety_score': safety_result['safety_score'],
        'max_safety_score': safety_result['max_safety_score'],
        
        # Individual safety scores
        'attribution_score': safety_result['attribution_score'],
        'consistency_score': safety_result['consistency_score'],
        'semantic_entropy': safety_result['semantic_entropy'],
        
        # Weak sentences
        'weak_sentences': safety_result['weak_sentences'],
        'has_weak_sentences': len(safety_result['weak_sentences']) > 0,
        
        # Configuration
        'multi_stage_used': safety_result['use_multi_stage'],
        'fact_check_enabled': safety_result['external_fact_check_enabled']
    }
    
    # Add external fact-checking results if available
    fact_check_result = safety_result.get('fact_check_result')
    if fact_check_result and not fact_check_result.get('error'):
        response['external_validation'] = {
            'score': fact_check_result.get('combined_score', 0.0),
            'num_sources': fact_check_result.get('num_external_sources', 0),
            'query_used': fact_check_result.get('query_used', ''),
            'interpretation': fact_check_result.get('external_interpretation', {}),
            'recommendation': fact_check_result.get('recommendation', '')
        }
    elif fact_check_result and fact_check_result.get('error'):
        response['external_validation'] = {
            'error': fact_check_result['error'],
            'score': 0.0
        }
    else:
        response['external_validation'] = None
    
    # Add safety interpretations
    response['safety_interpretations'] = get_safety_interpretations(safety_result)
    
    return response


def get_safety_interpretations(safety_result):
    """Get human-readable interpretations of safety scores."""
    interpretations = {}
    
    # Attribution interpretation
    attr_score = safety_result['attribution_score']
    if attr_score >= 0.7:
        interpretations['attribution'] = "Excellent - answer is well-grounded in sources"
    elif attr_score >= 0.6:
        interpretations['attribution'] = "Good - answer is mostly supported by sources"
    elif attr_score >= 0.4:
        interpretations['attribution'] = "Fair - some parts may lack source support"
    else:
        interpretations['attribution'] = "Poor - answer may contain unsupported claims"
    
    # Consistency interpretation
    cons_score = safety_result['consistency_score']
    if cons_score >= 0.8:
        interpretations['consistency'] = "High - very stable responses"
    elif cons_score >= 0.6:
        interpretations['consistency'] = "Good - mostly consistent responses"
    else:
        interpretations['consistency'] = "Low - responses vary significantly"
    
    # Entropy interpretation
    entropy = safety_result['semantic_entropy']
    if entropy < 1.0:
        interpretations['entropy'] = "Low uncertainty - confident answer"
    elif entropy < 2.0:
        interpretations['entropy'] = "Medium uncertainty - review recommended"
    else:
        interpretations['entropy'] = "High uncertainty - likely hallucination"
    
    return interpretations


if __name__ == "__main__":
    import uvicorn
    
    # Development configuration
    debug_mode = os.getenv("FASTAPI_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 8000))
    
    print(f"🌐 Starting healthcare RAG web interface...")
    print(f"   Debug mode: {debug_mode}")
    print(f"   Port: {port}")
    print(f"   Open: http://localhost:{port}")
    print(f"   API Docs: http://localhost:{port}/api/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=debug_mode,
        log_level="info"
    )