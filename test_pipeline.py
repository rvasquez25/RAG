"""
Testing and Validation Utilities for RAG Pipeline
Verify setup and measure retrieval quality
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple
import pymysql
from pathlib import Path


def test_singlestore_connection(config: Dict[str, Any]) -> bool:
    """
    Test connection to SingleStore
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        True if connection successful
    """
    print("\n=== Testing SingleStore Connection ===")
    try:
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        
        # Check version
        cursor.execute("SELECT @@version")
        version = cursor.fetchone()[0]
        print(f"✓ Connected to SingleStore")
        print(f"  Version: {version}")
        
        # Check vector support
        if '8.9' in version or '8.10' in version or any(f'8.{i}' in version for i in range(9, 20)):
            print(f"✓ Vector operations supported")
        else:
            print(f"⚠ Warning: Vector support requires SingleStore 8.9+")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_model_loading() -> bool:
    """
    Test loading of BGE-M3 model
    
    Returns:
        True if models load successfully
    """
    print("\n=== Testing Model Loading ===")
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading BGE-M3 embedder...")
        model = SentenceTransformer("BAAI/bge-m3")
        dim = model.get_sentence_embedding_dimension()
        print(f"✓ BGE-M3 loaded successfully")
        print(f"  Embedding dimension: {dim}")
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✓ Embedding test successful")
        print(f"  Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install sentence-transformers: pip install sentence-transformers")
        print("2. Check internet connection for model download")
        print("3. Ensure sufficient disk space (~2GB for BGE-M3)")
        return False


def test_reranker_loading() -> bool:
    """
    Test loading of reranker model
    
    Returns:
        True if reranker loads successfully
    """
    print("\n=== Testing Reranker Loading ===")
    try:
        from sentence_transformers import CrossEncoder
        
        print("Loading BGE-Reranker-v2-M3...")
        model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        print(f"✓ Reranker loaded successfully")
        
        # Test reranking
        pairs = [["query", "document 1"], ["query", "document 2"]]
        scores = model.predict(pairs)
        print(f"✓ Reranking test successful")
        print(f"  Score shape: {len(scores)}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Reranker loading failed: {e}")
        print("  Note: Reranker is optional but recommended for max accuracy")
        return False


def test_pdf_extraction() -> bool:
    """
    Test PDF extraction capabilities
    
    Returns:
        True if PDF extraction works
    """
    print("\n=== Testing PDF Extraction ===")
    try:
        import PyPDF2
        print("✓ PyPDF2 available")
        
        # Try to find a sample PDF
        sample_pdfs = list(Path(".").glob("*.pdf"))
        if sample_pdfs:
            test_pdf = sample_pdfs[0]
            print(f"  Testing with: {test_pdf.name}")
            
            with open(test_pdf, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                print(f"✓ PDF extraction successful")
                print(f"  Pages: {num_pages}")
                
                # Test first page
                text = reader.pages[0].extract_text()
                print(f"  First page text length: {len(text)} chars")
        else:
            print("  No sample PDFs found for testing")
            print("  Create a test PDF to verify extraction")
        
        return True
        
    except Exception as e:
        print(f"✗ PDF extraction failed: {e}")
        return False


def test_embeddings_storage(config: Dict[str, Any]) -> bool:
    """
    Test end-to-end embedding storage and retrieval
    
    Args:
        config: Database configuration
        
    Returns:
        True if storage/retrieval works
    """
    print("\n=== Testing Embeddings Storage ===")
    try:
        from rag_pipeline import BGEEmbedder, SingleStoreVectorDB, Document
        import numpy as np
        
        # Initialize
        embedder = BGEEmbedder("BAAI/bge-m3")
        vector_db = SingleStoreVectorDB(**config)
        
        # Create test table
        table_name = "test_embeddings"
        print(f"Creating test table: {table_name}")
        vector_db.create_table(table_name, embedder.embedding_dim)
        print("✓ Table created")
        
        # Create test documents
        test_docs = [
            Document(
                content="The quick brown fox jumps over the lazy dog.",
                source="test.txt",
                page_num=1,
                chunk_id="test_chunk_1",
                metadata={"test": True}
            ),
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                source="test.txt",
                page_num=1,
                chunk_id="test_chunk_2",
                metadata={"test": True}
            )
        ]
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [doc.content for doc in test_docs]
        embeddings = embedder.embed_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Store
        print("Storing embeddings...")
        vector_db.insert_embeddings(table_name, test_docs, embeddings)
        print("✓ Embeddings stored")
        
        # Test retrieval
        print("Testing retrieval...")
        query_text = "What is machine learning?"
        query_embedding = embedder.embed_text(query_text)
        results = vector_db.search(table_name, query_embedding, top_k=2)
        
        if results:
            print(f"✓ Retrieved {len(results)} results")
            print(f"  Top score: {results[0]['similarity_score']:.4f}")
            print(f"  Content: {results[0]['content'][:50]}...")
        else:
            print("⚠ No results returned")
            return False
        
        # Cleanup
        print("Cleaning up test table...")
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        cursor.close()
        conn.close()
        print("✓ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Storage test failed: {e}")
        return False


def benchmark_retrieval_speed(config: Dict[str, Any], num_queries: int = 10):
    """
    Benchmark retrieval speed
    
    Args:
        config: Database configuration
        num_queries: Number of test queries
    """
    print("\n=== Benchmarking Retrieval Speed ===")
    try:
        from rag_pipeline import BGEEmbedder, SingleStoreVectorDB
        
        embedder = BGEEmbedder("BAAI/bge-m3")
        vector_db = SingleStoreVectorDB(**config)
        
        # Check if we have any data
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            print("⚠ No tables found. Ingest some documents first.")
            cursor.close()
            conn.close()
            return
        
        table_name = tables[0][0]
        print(f"Using table: {table_name}")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Documents in table: {count}")
        cursor.close()
        conn.close()
        
        if count == 0:
            print("⚠ Table is empty. Ingest some documents first.")
            return
        
        # Benchmark queries
        test_queries = [
            "What is the main topic?",
            "Explain the methodology",
            "What are the conclusions?",
            "Describe the results",
            "What are the limitations?"
        ] * (num_queries // 5)
        
        times = []
        for i, query in enumerate(test_queries[:num_queries], 1):
            start = time.time()
            query_embedding = embedder.embed_text(query)
            results = vector_db.search(table_name, query_embedding, top_k=5)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i == 1:
                print(f"\nFirst query details:")
                print(f"  Query: {query}")
                print(f"  Time: {elapsed*1000:.1f}ms")
                print(f"  Results: {len(results)}")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nBenchmark Results ({num_queries} queries):")
        print(f"  Average: {avg_time*1000:.1f}ms")
        print(f"  Min: {min_time*1000:.1f}ms")
        print(f"  Max: {max_time*1000:.1f}ms")
        print(f"  Throughput: {1/avg_time:.1f} queries/second")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")


def validate_retrieval_quality(config: Dict[str, Any]):
    """
    Validate retrieval quality with sample queries
    
    Args:
        config: Database configuration
    """
    print("\n=== Validating Retrieval Quality ===")
    try:
        from advanced_rag import create_production_pipeline
        
        pipeline = create_production_pipeline(
            config,
            use_reranker=True,
            use_hybrid=True
        )
        
        # Check if we have data
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not tables:
            print("⚠ No data available. Ingest documents first.")
            return
        
        # Test queries
        test_cases = [
            {
                "query": "main topic",
                "expected_min_score": 0.3,
                "description": "Generic topic query"
            },
            {
                "query": "specific technical term that probably doesn't exist",
                "expected_min_score": 0.0,
                "description": "Query with no matches"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test['description']}")
            print(f"Query: '{test['query']}'")
            
            results = pipeline.query(
                test['query'],
                top_k=3,
                use_reranking=True
            )
            
            if results:
                top_score = results[0].get('rerank_score', results[0].get('similarity_score'))
                print(f"  Results found: {len(results)}")
                print(f"  Top score: {top_score:.4f}")
                
                if top_score >= test['expected_min_score']:
                    print(f"  ✓ Score meets expectations (>= {test['expected_min_score']})")
                else:
                    print(f"  ⚠ Score below expectations (< {test['expected_min_score']})")
            else:
                print(f"  No results found")
                if test['expected_min_score'] == 0.0:
                    print(f"  ✓ Expected behavior")
                else:
                    print(f"  ⚠ Expected some results")
        
    except Exception as e:
        print(f"✗ Quality validation failed: {e}")


def run_all_tests(config: Dict[str, Any]):
    """
    Run complete test suite
    
    Args:
        config: Database configuration
    """
    print("\n" + "="*60)
    print("RAG PIPELINE TEST SUITE")
    print("="*60)
    
    results = {
        "Connection": test_singlestore_connection(config),
        "Model Loading": test_model_loading(),
        "Reranker Loading": test_reranker_loading(),
        "PDF Extraction": test_pdf_extraction(),
        "Storage": test_embeddings_storage(config)
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Your setup is ready.")
        
        # Run optional benchmarks
        print("\n" + "="*60)
        benchmark_retrieval_speed(config, num_queries=10)
        validate_retrieval_quality(config)
        
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


def main():
    """Main test runner"""
    
    # Load configuration
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    print("Configuration:")
    print(f"  Host: {config['host']}")
    print(f"  Port: {config['port']}")
    print(f"  Database: {config['database']}")
    
    # Run tests
    run_all_tests(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
