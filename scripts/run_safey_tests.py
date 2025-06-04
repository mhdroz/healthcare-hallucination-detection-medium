#!/usr/bin/env python3
"""
CLI script to test safety checks on RAG system.
Based on blog post 3 code.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag import create_query_engine
from safety import comprehensive_safety_check
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


def main():
    """Test safety checks on RAG system."""
    parser = argparse.ArgumentParser(description="Test safety checks on RAG system")
    
    parser.add_argument("--index-path", type=str, required=True, help="Path to saved index")
    parser.add_argument("--question", type=str, help="Question to test (optional)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", 
                       help="Embedding model used")
    parser.add_argument("--encoder-model", type=str, 
                       default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                       help="Encoder model for safety checks")
    parser.add_argument("--multi-stage", action="store_true", help="Use multi-stage retrieval")
    parser.add_argument("--consistency-tries", type=int, default=3, 
                       help="Number of consistency check attempts")
    parser.add_argument("--disable-fact-check", action="store_true", 
                       help="Disable external fact-checking")
    parser.add_argument("--max-external-sources", type=int, default=10,
                       help="Maximum external sources for fact-checking")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for LLM")
        return
    
    # Default test question from blog post 3
    test_question = args.question or "What antibiotics are safe to use with warfarin in elderly patients?"
    
    print(f"🛡️  Testing safety checks:")
    print(f"   Index: {args.index_path}")
    print(f"   Question: {test_question}")
    print(f"   Multi-stage: {args.multi_stage}")
    print(f"   Consistency tries: {args.consistency_tries}")
    print(f"   External fact-check: {not args.disable_fact_check}")
    print("="*70)
    
    try:
        # Setup models
        embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        encoder = SentenceTransformer(args.encoder_model)
        
        # Create query engine
        query_engine = create_query_engine(args.index_path, llm, embed_model)
        
        # Run comprehensive safety check
        safety_result = comprehensive_safety_check(
            question=test_question,
            query_engine=query_engine,
            llm=llm,
            encoder=encoder,
            num_tries=args.consistency_tries,
            use_multi_stage=args.multi_stage,
            enable_fact_check=not args.disable_fact_check
        )
        
        # Additional summary
        print(f"\n📋 SAFETY SUMMARY")
        print("="*30)
        print(f"Final Confidence: {safety_result['confidence']}")
        print(f"Safety Score: {safety_result['safety_score']}/{safety_result['max_safety_score']}")
        print(f"Weak Sentences: {len(safety_result['weak_sentences'])}")
        
        # External fact-checking summary
        if not args.disable_fact_check and safety_result.get('fact_check_result'):
            fact_result = safety_result['fact_check_result']
            if not fact_result.get('error'):
                print(f"External Sources: {fact_result.get('num_external_sources', 0)}")
                print(f"External Score: {fact_result.get('combined_score', 0.0):.3f}")
            else:
                print(f"External Check: Failed ({fact_result['error']})")
        
        if safety_result['confidence'] == "LOW CONFIDENCE":
            print(f"\n⚠️  WARNING: This response may contain hallucinations!")
            print(f"   Consider improving the corpus or adjusting parameters.")
        elif safety_result['confidence'] == "MEDIUM CONFIDENCE":
            print(f"\n⚠️  CAUTION: Review this response carefully.")
        else:
            print(f"\n✅ Response appears safe and well-grounded.")
        
        print(f"\n💡 Tips to improve safety:")
        print(f"   • Expand corpus with more relevant articles")
        print(f"   • Use biomedical embedding models")
        print(f"   • Add reranking to retrieval")
        print(f"   • Try multi-stage retrieval for complex queries")
        
    except Exception as e:
        print(f"❌ Error during safety testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()