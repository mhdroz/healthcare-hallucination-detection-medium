from fastapi import APIRouter, Request, HTTPException, FastAPI
from fastapi.responses import RedirectResponse

from .models import SafetyResponse, QueryRequest, EvaluationResponse, HealthResponse, EvaluationRequest
from .utils import _sanitize_numpy_types, format_safety_response, debug_print
from safety import comprehensive_safety_check
from rag import run_full_evaluation, create_query_engine
import config as cfg
from llama_index.llms.openai import OpenAI




router = APIRouter()

@router.get("/")
async def root(request: Request):
    """Redirect to API documentation."""
    return RedirectResponse(url="/api/docs")



@router.post("/api/query", response_model=SafetyResponse)
async def handle_query(query_request: QueryRequest, request: Request):
    """Handle medical queries with full safety analysis."""
    try:
        #query_engine = request.app.state.query_engine
        index = request.app.state.index
        llm = request.app.state.llm
        encoder = request.app.state.encoder
        embed_model = request.app.state.embed_model
        query_engine = create_query_engine(index, llm, embed_model, query_request.use_reranker)
        

        if not query_engine:
            raise HTTPException(status_code=503, detail="Models not initialized")
        
        debug_print(f"Processing query: {query_request.question}")
        
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
        
        # Sanitize the entire result for any NumPy numeric types before further processing
        safety_result = _sanitize_numpy_types(safety_result)
        
        # Format response
        response = format_safety_response(safety_result)
        return SafetyResponse(**response)
        
    except Exception as e:
        debug_print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/evaluate", response_model=EvaluationResponse)
async def handle_evaluation(eval_request: EvaluationRequest, request: Request):
    """Run RAGAS evaluation on the system."""
    try:
        #query_engine = request.app.state.query_engine
        index = request.app.state.index
        llm = request.app.state.llm
        embed_model = request.app.state.embed_model
        query_engine = create_query_engine(index, llm, embed_model, eval_request.use_reranker)
        
        if not query_engine:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        debug_print("Running RAGAS evaluation...")
        
        judge_llm = OpenAI(model=cfg.OPENAI_MODEL_NAME, temperature=cfg.DEFAULT_TEMPERATURE)
        
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
        debug_print(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Simple health check endpoint."""
    # Check if all required models are loaded
    models_loaded = all([
        hasattr(request.app.state, 'index') and request.app.state.index is not None,
        hasattr(request.app.state, 'llm') and request.app.state.llm is not None,
        hasattr(request.app.state, 'encoder') and request.app.state.encoder is not None,
        hasattr(request.app.state, 'embed_model') and request.app.state.embed_model is not None
    ])
    print(f"Models loaded: {models_loaded}")
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        models_loaded=models_loaded
    )


@router.get("/api/sample-questions")
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
