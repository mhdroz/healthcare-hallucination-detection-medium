"""
RAGAS evaluation for medical RAG system.
Based on blog post 2 evaluation code.
"""
from typing import List, Dict
from ragas.llms import LlamaIndexLLMWrapper
from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.integrations.llama_index import evaluate as llama_evaluate
import pandas as pd


def create_pneumonia_test_questions() -> List[str]:
    """Create test questions from blog post 2."""
    return [
        # community-acquired pneumonia (CAP)
        "What is the first-line antibiotic regimen for outpatient treatment of uncomplicated community-acquired bacterial pneumonia in an adult with no comorbidities?",
        "When is dual therapy with a β-lactam plus macrolide preferred over monotherapy for CAP?",
        "Recommended duration of therapy for uncomplicated CAP caused by Streptococcus pneumoniae?",
        "How does recommended empiric therapy change for CAP in regions with >25% macrolide-resistant S. pneumoniae?",
        "Which respiratory fluoroquinolones are acceptable alternatives for CAP in a patient with severe penicillin allergy?",
        
        # hospital-acquired / ventilator-associated pneumonia (HAP/VAP)
        "What empiric coverage is advised for hospital-acquired pneumonia when MRSA risk factors are present?",
        "First-line IV therapy for severe CAP requiring ICU admission with pseudomonal risk?",
        "Role of local antibiogram data in selecting empiric therapy for ventilator-associated pneumonia?",
        
        # special situations
        "When should azithromycin dose be adjusted in moderate renal impairment?",
        "Preferred outpatient therapy for CAP in a pregnant patient during the second trimester?",
        "How does the guideline differ for treating aspiration pneumonia with anaerobic coverage?",
        "Recommended approach if a patient remains febrile after 48 h of appropriate CAP therapy?",
    ]


def create_evaluation_dataset(questions: List[str]) -> EvaluationDataset:
    """Create RAGAS evaluation dataset from questions (from blog post)."""
    samples = [
        SingleTurnSample(
            id=f"q{i}",
            user_input=q,  # the question
            answer=None,   # use real answers if you have them
            contexts=[],   # and real citational chunks if you have them
        )
        for i, q in enumerate(questions)
    ]
    
    eval_ds = EvaluationDataset(samples)
    return eval_ds


def evaluate_rag_system(query_engine, judge_llm, questions: List[str] = None) -> Dict:
    """
    Evaluate RAG system using RAGAS metrics (from blog post 2).
    
    Args:
        query_engine: LlamaIndex query engine to evaluate
        judge_llm: LLM to use as judge (e.g., GPT-4)
        questions: List of questions to evaluate (optional)
    
    Returns:
        Evaluation results with scores
    """
    # Use default pneumonia questions if none provided
    if questions is None:
        questions = create_pneumonia_test_questions()
    
    print(f"🧪 Evaluating RAG system with {len(questions)} questions")
    
    # Create evaluation dataset
    eval_ds = create_evaluation_dataset(questions)
    
    # Wrap the judge LLM for RAGAS
    judge = LlamaIndexLLMWrapper(judge_llm)
    
    # Define metrics (from blog post)
    metrics = [Faithfulness(llm=judge), AnswerRelevancy(llm=judge)]
    
    # Run evaluation
    print("🔍 Running RAGAS evaluation...")
    scores = llama_evaluate(query_engine=query_engine, metrics=metrics, dataset=eval_ds)
    
    print("✅ Evaluation complete!")
    return scores


def get_evaluation_summary(scores) -> Dict:
    """Get summary statistics from RAGAS evaluation."""
    # Convert to pandas for easier analysis
    scores_df = scores.to_pandas()
    
    summary = {
        "num_questions": len(scores_df),
        "faithfulness_mean": scores_df["faithfulness"].mean(),
        "faithfulness_std": scores_df["faithfulness"].std(),
        "answer_relevancy_mean": scores_df["answer_relevancy"].mean(), 
        "answer_relevancy_std": scores_df["answer_relevancy"].std(),
        "detailed_scores": scores_df.to_dict('records')
    }
    
    return summary


def interpret_scores(summary: Dict) -> Dict:
    """Interpret RAGAS scores with explanations (from blog post)."""
    faithfulness = summary["faithfulness_mean"]
    relevancy = summary["answer_relevancy_mean"]
    
    # Interpretation from blog post
    if faithfulness >= 0.8:
        faith_interp = "Excellent - answers are well grounded in sources"
    elif faithfulness >= 0.6:
        faith_interp = "Good - most answers are grounded in sources"
    elif faithfulness >= 0.4:
        faith_interp = "Fair - some answers may contain hallucinations"
    else:
        faith_interp = "Poor - significant hallucination risk"
    
    if relevancy >= 0.8:
        rel_interp = "Excellent - answers are highly relevant to questions"
    elif relevancy >= 0.6:
        rel_interp = "Good - answers are mostly relevant"
    elif relevancy >= 0.4:
        rel_interp = "Fair - answers are somewhat relevant"
    else:
        rel_interp = "Poor - answers may not address the questions well"
    
    return {
        "faithfulness_score": faithfulness,
        "faithfulness_interpretation": faith_interp,
        "relevancy_score": relevancy,
        "relevancy_interpretation": rel_interp,
        "overall_grade": _get_overall_grade(faithfulness, relevancy)
    }


def _get_overall_grade(faithfulness: float, relevancy: float) -> str:
    """Get overall system grade."""
    avg_score = (faithfulness + relevancy) / 2
    
    if avg_score >= 0.8:
        return "A - Excellent RAG system"
    elif avg_score >= 0.7:
        return "B - Good RAG system" 
    elif avg_score >= 0.6:
        return "C - Fair RAG system"
    elif avg_score >= 0.5:
        return "D - Poor RAG system"
    else:
        return "F - Failing RAG system"


def plot_evaluation_results(scores):
    """Create scatter plot from blog post 2."""
    try:
        import matplotlib.pyplot as plt
        
        scores_df = scores.to_pandas()
        
        plt.figure(figsize=(8, 6))
        plt.scatter(scores_df["faithfulness"], scores_df["answer_relevancy"], alpha=0.7)
        plt.xlabel("Faithfulness (0-1)")
        plt.ylabel("Answer Relevancy (0-1)")
        plt.title("Per-question RAGAS Scores")
        plt.grid(True, linestyle="--", alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib not available for plotting")


def run_full_evaluation(query_engine, judge_llm, questions: List[str] = None, 
                       show_plot: bool = True) -> Dict:
    """
    Run complete evaluation pipeline (convenience function).
    
    Args:
        query_engine: Query engine to evaluate
        judge_llm: Judge LLM for evaluation
        questions: Questions to test (optional)
        show_plot: Whether to show the scatter plot
    
    Returns:
        Complete evaluation results
    """
    # Run RAGAS evaluation
    scores = evaluate_rag_system(query_engine, judge_llm, questions)
    
    # Get summary statistics
    summary = get_evaluation_summary(scores)
    
    # Interpret results
    interpretation = interpret_scores(summary)
    
    # Print results
    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Questions evaluated: {summary['num_questions']}")
    print(f"Faithfulness: {interpretation['faithfulness_score']:.3f} - {interpretation['faithfulness_interpretation']}")
    print(f"Answer Relevancy: {interpretation['relevancy_score']:.3f} - {interpretation['relevancy_interpretation']}")
    print(f"Overall Grade: {interpretation['overall_grade']}")
    
    # Show plot if requested
    if show_plot:
        plot_evaluation_results(scores)
    
    return {
        "scores": scores,
        "summary": summary,
        "interpretation": interpretation
    }