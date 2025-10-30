"""
RAG Evaluation System
Tests RAG pipeline against ground truth questions and answers
"""

import os
import yaml
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import time
from datetime import datetime
import logging

# Import RAG components
from rag_pipeline import BGEEmbedder, SingleStoreVectorDB
from advanced_rag import create_production_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    expected_response: str
    actual_response: str
    retrieved_chunks: List[Dict[str, Any]]
    similarity_score: float
    exact_match: bool
    semantic_match: bool
    retrieval_time_ms: float
    contains_expected: bool
    top_k_used: int


@dataclass
class EvaluationSummary:
    """Overall evaluation summary"""
    total_questions: int
    exact_matches: int
    semantic_matches: int
    contains_expected: int
    avg_similarity: float
    avg_retrieval_time_ms: float
    pass_rate: float
    results: List[EvaluationResult]


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance against ground truth Q&A pairs
    """
    
    def __init__(
        self,
        pipeline,
        semantic_threshold: float = 0.7,
        use_reranking: bool = True
    ):
        """
        Initialize evaluator
        
        Args:
            pipeline: AdvancedRAGPipeline instance
            semantic_threshold: Minimum similarity for semantic match
            use_reranking: Whether to use reranking in queries
        """
        self.pipeline = pipeline
        self.semantic_threshold = semantic_threshold
        self.use_reranking = use_reranking
    
    def load_questions_from_yaml(self, yaml_path: str) -> List[Dict[str, str]]:
        """
        Load questions and expected responses from YAML file
        
        Supported formats:
        1. Simple list:
           - question: "What is X?"
             expectedResponse: "Y"
        
        2. Nested format:
           questions:
             - question: "What is X?"
               expectedResponse: "Y"
        
        3. Alternative field names (q/a, answer, expected_response)
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            List of dicts with 'question' and 'expectedResponse' keys
            
        Raises:
            ValueError: If YAML format is invalid or missing required fields
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        
        if not data:
            raise ValueError("YAML file is empty")
        
        # Handle different YAML formats
        if isinstance(data, list):
            # Format: list of dicts with question/expectedResponse
            questions = data
        elif isinstance(data, dict) and 'questions' in data:
            # Format: {questions: [...]}
            questions = data['questions']
        else:
            raise ValueError(
                "YAML format not recognized. Expected:\n"
                "- List of question/answer dicts\n"
                "- Dict with 'questions' key containing a list\n\n"
                "Example:\n"
                "- question: \"What is X?\"\n"
                "  expectedResponse: \"Y\""
            )
        
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")
        
        if not questions:
            raise ValueError("Questions list is empty")
        
        # Normalize field names and validate
        normalized = []
        for i, item in enumerate(questions, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Question {i} is not a dict: {item}")
            
            # Extract question
            question = (
                item.get('question') or 
                item.get('q') or 
                item.get('query')
            )
            
            # Extract expected response
            expected = (
                item.get('expectedResponse') or 
                item.get('expected_response') or 
                item.get('answer') or 
                item.get('a') or
                item.get('expected')
            )
            
            # Validate required fields
            if not question:
                raise ValueError(
                    f"Question {i} missing 'question' field. "
                    f"Available keys: {list(item.keys())}"
                )
            
            if not expected:
                raise ValueError(
                    f"Question {i} missing 'expectedResponse' field. "
                    f"Available keys: {list(item.keys())}"
                )
            
            normalized.append({
                'question': str(question).strip(),
                'expectedResponse': str(expected).strip()
            })
        
        logger.info(f"Loaded {len(normalized)} questions from {yaml_path}")
        return normalized
    
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts
        Uses the same embedder as the RAG pipeline
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        emb1 = self.pipeline.embedder.embed_text(text1)
        emb2 = self.pipeline.embedder.embed_text(text2)
        
        # Cosine similarity (embeddings are normalized)
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)
    
    def check_exact_match(
        self,
        expected: str,
        actual: str,
        case_sensitive: bool = False
    ) -> bool:
        """Check if texts match exactly"""
        if not case_sensitive:
            expected = expected.lower().strip()
            actual = actual.lower().strip()
        else:
            expected = expected.strip()
            actual = actual.strip()
        
        return expected == actual
    
    def check_contains(
        self,
        expected: str,
        actual: str,
        case_sensitive: bool = False
    ) -> bool:
        """Check if actual text contains expected text"""
        if not case_sensitive:
            expected = expected.lower().strip()
            actual = actual.lower().strip()
        else:
            expected = expected.strip()
            actual = actual.strip()
        
        return expected in actual
    
    def evaluate_single(
        self,
        question: str,
        expected_response: str,
        top_k: int = 5,
        retrieval_k: int = 20
    ) -> EvaluationResult:
        """
        Evaluate a single question
        
        Args:
            question: Question to ask
            expected_response: Expected answer
            top_k: Number of final results
            retrieval_k: Number of candidates for reranking
            
        Returns:
            EvaluationResult object
        """
        # Query the RAG system
        start_time = time.time()
        
        results = self.pipeline.query(
            question,
            top_k=top_k,
            retrieval_k=retrieval_k,
            use_reranking=self.use_reranking
        )
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Concatenate retrieved content
        actual_response = "\n".join([r['content'] for r in results])
        
        # Compute metrics
        exact_match = self.check_exact_match(expected_response, actual_response)
        contains_expected = self.check_contains(expected_response, actual_response)
        
        # Semantic similarity
        similarity_score = self.compute_semantic_similarity(
            expected_response,
            actual_response
        )
        semantic_match = similarity_score >= self.semantic_threshold
        
        return EvaluationResult(
            question=question,
            expected_response=expected_response,
            actual_response=actual_response,
            retrieved_chunks=results,
            similarity_score=similarity_score,
            exact_match=exact_match,
            semantic_match=semantic_match,
            retrieval_time_ms=retrieval_time_ms,
            contains_expected=contains_expected,
            top_k_used=len(results)
        )
    
    def evaluate_all(
        self,
        yaml_path: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        verbose: bool = True
    ) -> EvaluationSummary:
        """
        Evaluate all questions from YAML file
        
        Args:
            yaml_path: Path to YAML file with questions
            top_k: Number of final results per query
            retrieval_k: Number of candidates for reranking
            verbose: Print progress
            
        Returns:
            EvaluationSummary object
        """
        # Load questions
        questions = self.load_questions_from_yaml(yaml_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RAG EVALUATION")
            print(f"{'='*60}")
            print(f"Questions: {len(questions)}")
            print(f"Top-K: {top_k}")
            print(f"Retrieval-K: {retrieval_k}")
            print(f"Reranking: {self.use_reranking}")
            print(f"Semantic Threshold: {self.semantic_threshold}")
            print(f"{'='*60}\n")
        
        results = []
        
        for i, qa in enumerate(questions, 1):
            if verbose:
                print(f"[{i}/{len(questions)}] Evaluating: {qa['question'][:60]}...")
            
            try:
                result = self.evaluate_single(
                    qa['question'],
                    qa['expectedResponse'],
                    top_k=top_k,
                    retrieval_k=retrieval_k
                )
                results.append(result)
                
                if verbose:
                    status = "[PASS]" if result.semantic_match else "[FAIL]"
                    print(f"  {status} Similarity: {result.similarity_score:.3f} | "
                          f"Contains: {result.contains_expected} | "
                          f"Time: {result.retrieval_time_ms:.1f}ms")
                
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] Error: {e}")
                # Create failed result
                results.append(EvaluationResult(
                    question=qa['question'],
                    expected_response=qa['expectedResponse'],
                    actual_response="",
                    retrieved_chunks=[],
                    similarity_score=0.0,
                    exact_match=False,
                    semantic_match=False,
                    retrieval_time_ms=0.0,
                    contains_expected=False,
                    top_k_used=0
                ))
        
        # Compute summary
        total = len(results)
        exact_matches = sum(1 for r in results if r.exact_match)
        semantic_matches = sum(1 for r in results if r.semantic_match)
        contains_expected = sum(1 for r in results if r.contains_expected)
        avg_similarity = sum(r.similarity_score for r in results) / total if total > 0 else 0
        avg_time = sum(r.retrieval_time_ms for r in results) / total if total > 0 else 0
        
        # Pass rate based on semantic matches
        pass_rate = semantic_matches / total if total > 0 else 0
        
        summary = EvaluationSummary(
            total_questions=total,
            exact_matches=exact_matches,
            semantic_matches=semantic_matches,
            contains_expected=contains_expected,
            avg_similarity=avg_similarity,
            avg_retrieval_time_ms=avg_time,
            pass_rate=pass_rate,
            results=results
        )
        
        if verbose:
            self.print_summary(summary)
        
        return summary
    
    def print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Questions:     {summary.total_questions}")
        print(f"Exact Matches:       {summary.exact_matches} "
              f"({summary.exact_matches/summary.total_questions*100:.1f}%)")
        print(f"Semantic Matches:    {summary.semantic_matches} "
              f"({summary.pass_rate*100:.1f}%)")
        print(f"Contains Expected:   {summary.contains_expected} "
              f"({summary.contains_expected/summary.total_questions*100:.1f}%)")
        print(f"Avg Similarity:      {summary.avg_similarity:.3f}")
        print(f"Avg Retrieval Time:  {summary.avg_retrieval_time_ms:.1f}ms")
        print(f"{'='*60}")
        
        # Pass/Fail
        if summary.pass_rate >= 0.8:
            print("[PASS] Pass rate >= 80% (semantic matches)")
        elif summary.pass_rate >= 0.6:
            print("[MARGINAL] Pass rate 60-80% (semantic matches)")
        else:
            print("[FAIL] Pass rate < 60% (semantic matches)")
        print(f"{'='*60}\n")
    
    def save_results(
        self,
        summary: EvaluationSummary,
        output_path: str,
        format: str = 'json'
    ):
        """
        Save evaluation results to file
        
        Args:
            summary: EvaluationSummary object
            output_path: Path to save results
            format: 'json' or 'yaml'
        """
        # Convert to serializable format
        data = {
            'evaluation_date': datetime.now().isoformat(),
            'summary': {
                'total_questions': summary.total_questions,
                'exact_matches': summary.exact_matches,
                'semantic_matches': summary.semantic_matches,
                'contains_expected': summary.contains_expected,
                'avg_similarity': summary.avg_similarity,
                'avg_retrieval_time_ms': summary.avg_retrieval_time_ms,
                'pass_rate': summary.pass_rate
            },
            'results': []
        }
        
        for result in summary.results:
            data['results'].append({
                'question': result.question,
                'expected_response': result.expected_response,
                'actual_response': result.actual_response[:500] + '...' if len(result.actual_response) > 500 else result.actual_response,
                'similarity_score': result.similarity_score,
                'exact_match': result.exact_match,
                'semantic_match': result.semantic_match,
                'contains_expected': result.contains_expected,
                'retrieval_time_ms': result.retrieval_time_ms,
                'top_k_used': result.top_k_used,
                'retrieved_sources': [
                    {
                        'source': chunk['source'],
                        'page': chunk.get('page_num'),
                        'score': chunk.get('rerank_score') or chunk.get('similarity_score')
                    }
                    for chunk in result.retrieved_chunks
                ]
            })
        
        # Save
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Results saved to: {output_path}")
    
    def generate_report(
        self,
        summary: EvaluationSummary,
        output_path: str
    ):
        """
        Generate detailed HTML report
        
        Args:
            summary: EvaluationSummary object
            output_path: Path to save HTML report
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RAG Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 24px;
            color: #4CAF50;
        }}
        .pass {{
            color: #4CAF50;
        }}
        .fail {{
            color: #f44336;
        }}
        .marginal {{
            color: #ff9800;
        }}
        .result {{
            border: 1px solid #ddd;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .result.pass {{
            border-left: 4px solid #4CAF50;
        }}
        .result.fail {{
            border-left: 4px solid #f44336;
        }}
        .question {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .expected {{
            background: #e3f2fd;
            padding: 10px;
            border-radius: 3px;
            margin: 5px 0;
        }}
        .actual {{
            background: #f1f8e9;
            padding: 10px;
            border-radius: 3px;
            margin: 5px 0;
        }}
        .metrics {{
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-label">Total Questions</div>
                <div class="metric-value">{summary.total_questions}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Pass Rate</div>
                <div class="metric-value {'pass' if summary.pass_rate >= 0.8 else 'fail' if summary.pass_rate < 0.6 else 'marginal'}">
                    {summary.pass_rate*100:.1f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Similarity</div>
                <div class="metric-value">{summary.avg_similarity:.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Time</div>
                <div class="metric-value">{summary.avg_retrieval_time_ms:.0f}ms</div>
            </div>
        </div>
        
        <h2>Detailed Results</h2>
"""
        
        for i, result in enumerate(summary.results, 1):
            status_class = 'pass' if result.semantic_match else 'fail'
            status_icon = '[PASS]' if result.semantic_match else '[FAIL]'
            
            html += f"""
        <div class="result {status_class}">
            <div class="question">{status_icon} Question {i}: {result.question}</div>
            <div class="expected">
                <strong>Expected:</strong> {result.expected_response}
            </div>
            <div class="actual">
                <strong>Retrieved:</strong> {result.actual_response[:300]}{'...' if len(result.actual_response) > 300 else ''}
            </div>
            <div class="metrics">
                Similarity: {result.similarity_score:.3f} | 
                Contains Expected: {'Yes' if result.contains_expected else 'No'} | 
                Time: {result.retrieval_time_ms:.1f}ms | 
                Chunks: {result.top_k_used}
            </div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report saved to: {output_path}")


def main():
    """Example usage with better error handling"""
    
    # Configuration
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Get YAML file path
    yaml_path = "evaluation_questions.yml"
    
    # Check if evaluation file exists
    if not Path(yaml_path).exists():
        print(f"\nCreating example evaluation file: {yaml_path}")
        
        example_questions = [
            {
                "question": "What is the name of Singapore's tax authority?",
                "expectedResponse": "Inland Revenue Authority of Singapore"
            },
            {
                "question": "What is the IRAS website for Transfer Pricing?",
                "expectedResponse": "https://www.iras.gov.sg/taxes/corporate-income-tax/specific-topics/transfer-pricing"
            },
            {
                "question": "What does MLI stand for?",
                "expectedResponse": "Multilateral Instrument"
            }
        ]
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_questions, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[OK] Created example file: {yaml_path}")
        print("\nExample format:")
        print("- question: \"What is X?\"")
        print("  expectedResponse: \"Y\"")
        print("- question: \"Another question?\"")
        print("  expectedResponse: \"Another answer\"")
        print("\nNext steps:")
        print("1. Add your documents to the RAG system first")
        print("2. Update the questions in evaluation_questions.yml")
        print("3. Run this script again to evaluate")
        print("\nTo add documents, use:")
        print("  python example_usage.py")
        print("  # Or use your ingestion script")
        return
    
    print(f"\nLoading evaluation questions from: {yaml_path}")
    
    # Validate questions file
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            print("[ERROR] Evaluation file is empty")
            print(f"\nPlease add questions to {yaml_path}")
            print("\nExample format:")
            print("- question: \"What is X?\"")
            print("  expectedResponse: \"Y\"")
            return
        
        # Quick validation
        if isinstance(data, list):
            if len(data) == 0:
                print("[ERROR] No questions in file")
                return
            
            # Check first item
            first = data[0]
            if not isinstance(first, dict):
                print("[ERROR] Invalid format. Each item should be a dict.")
                return
            
            if 'question' not in first and 'q' not in first:
                print("[ERROR] Missing 'question' field")
                print(f"   Found keys: {list(first.keys())}")
                return
            
            if 'expectedResponse' not in first and 'answer' not in first and 'expected_response' not in first:
                print("[ERROR] Missing 'expectedResponse' field")
                print(f"   Found keys: {list(first.keys())}")
                print("\nExpected format:")
                print("- question: \"What is X?\"")
                print("  expectedResponse: \"Y\"")
                return
        
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML syntax: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Error reading file: {e}")
        return
    
    # Create pipeline
    print("\nInitializing RAG pipeline...")
    try:
        pipeline = create_production_pipeline(
            config,
            use_reranker=True,
            use_hybrid=True
        )
    except Exception as e:
        print(f"[ERROR] Error initializing pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure SingleStore is running")
        print("2. Check database exists: CREATE DATABASE rag_db;")
        print("3. Verify environment variables are set")
        print("4. Ensure documents have been ingested")
        return
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = RAGEvaluator(
        pipeline,
        semantic_threshold=0.7,
        use_reranking=True
    )
    
    # Run evaluation
    print("\nRunning evaluation...\n")
    try:
        summary = evaluator.evaluate_all(
            yaml_path,
            top_k=5,
            retrieval_k=20,
            verbose=True
        )
    except ValueError as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_path = f'evaluation_results_{timestamp}.json'
    evaluator.save_results(summary, json_path, format='json')
    
    html_path = f'evaluation_report_{timestamp}.html'
    evaluator.generate_report(summary, html_path)
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Evaluation Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")
    print(f"\nOpen {html_path} in your browser to view the detailed report.")
    
    # Print actionable insights
    if summary.pass_rate < 0.6:
        print(f"\n[WARNING] Low pass rate ({summary.pass_rate*100:.1f}%). Consider:")
        print("  1. Checking if documents are properly ingested")
        print("  2. Reviewing if questions match document content")
        print("  3. Adjusting chunk_size or retrieval parameters")
        print("  4. Lowering semantic_threshold (currently 0.7)")
    elif summary.pass_rate < 0.8:
        print(f"\n[INFO] Moderate pass rate ({summary.pass_rate*100:.1f}%). Room for improvement:")
        print("  1. Consider enabling reranking if not already enabled")
        print("  2. Try hybrid search for better recall")
        print("  3. Review failed questions for patterns")
    else:
        print(f"\n[SUCCESS] Great pass rate ({summary.pass_rate*100:.1f}%)! Your RAG system is performing well.")



if __name__ == "__main__":
    main()
