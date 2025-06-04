#!/usr/bin/env python3
"""
CLI script to evaluate RAG system using RAGAS.
Based on blog post 2 evaluation code.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag import create_query_engine, run_full_evaluation
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv


def main():
    """Evaluate RAG system with RAGAS."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system using RAGAS metrics")
    
    parser.add_argument("--index-path", type=str, required=True, help="Path to saved index")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Judge LLM model")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model used")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for evaluation judge")
        return
    
    print(f"📊 Evaluating RAG system:")
    print(f"   Index: {args.index_path}")
    print(f"   Judge model: {args.judge_model}")
    print(f"   Embedding model: {args.embedding_model}")
    print("="*60)
    
    try:
        # Setup models
        embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        judge_llm = OpenAI(model=args.judge_model, temperature=0.1)
        
        # Create query engine
        query_engine = create_query_engine(args.index_path, llm, embed_model)
        
        # Run full evaluation
        results = run_full_evaluation(
            query_engine=query_engine,
            judge_llm=judge_llm,
            show_plot=not args.no_plot
        )
        
        # Additional detailed output
        print(f"\n📈 Detailed Results:")
        summary = results["summary"]
        print(f"   Faithfulness: {summary['faithfulness_mean']:.3f} ± {summary['faithfulness_std']:.3f}")
        print(f"   Answer Relevancy: {summary['answer_relevancy_mean']:.3f} ± {summary['answer_relevancy_std']:.3f}")
        
        # Show worst performing questions
        detailed = summary["detailed_scores"]
        worst_faith = min(detailed, key=lambda x: x["faithfulness"])
        worst_rel = min(detailed, key=lambda x: x["answer_relevancy"])
        
        print(f"\n⚠️  Lowest scoring questions:")
        print(f"   Faithfulness ({worst_faith['faithfulness']:.3f}): Question {worst_faith['id']}")
        print(f"   Relevancy ({worst_rel['answer_relevancy']:.3f}): Question {worst_rel['id']}")
        
        print(f"\n✅ Evaluation complete! Use these scores to improve your RAG system.")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()