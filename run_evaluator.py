#!/usr/bin/env python3
"""
Simple RAG Evaluation Runner
Run this script to evaluate your RAG pipeline
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_evaluator import RAGEvaluator
from advanced_rag import create_production_pipeline


def main():
    """Run RAG evaluation"""
    
    # Configuration
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Get YAML file path
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else 'evaluation_questions.yml'
    
    if not Path(yaml_path).exists():
        print(f"Error: Evaluation file not found: {yaml_path}")
        print("\nUsage: python run_evaluation.py [path_to_questions.yml]")
        print("\nExample YAML format:")
        print("- question: \"What is X?\"")
        print("  expectedResponse: \"Y\"")
        print("\nSee evaluation_questions.yml for a complete example.")
        sys.exit(1)
    
    print(f"Loading evaluation questions from: {yaml_path}")
    
    # Create pipeline
    print("Initializing RAG pipeline...")
    try:
        pipeline = create_production_pipeline(
            config,
            use_reranker=True,
            use_hybrid=True
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nMake sure:")
        print("1. SingleStore is running")
        print("2. Database exists and has documents")
        print("3. Environment variables are set")
        sys.exit(1)
    
    # Create evaluator
    evaluator = RAGEvaluator(
        pipeline,
        semantic_threshold=0.7,
        use_reranking=True
    )
    
    # Run evaluation
    try:
        summary = evaluator.evaluate_all(
            yaml_path,
            top_k=5,
            retrieval_k=20,
            verbose=True
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    timestamp = Path(yaml_path).stem
    
    json_path = f'evaluation_results_{timestamp}.json'
    evaluator.save_results(summary, json_path, format='json')
    
    html_path = f'evaluation_report_{timestamp}.html'
    evaluator.generate_report(summary, html_path)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")
    print(f"\nOpen {html_path} in your browser to view the report.")
    
    # Exit with appropriate code
    sys.exit(0 if summary.pass_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
