"""
Complete End-to-End Example: High-Accuracy RAG Pipeline
Demonstrates ingestion, querying, and result analysis
"""

import os
import sys
from pathlib import Path

# Import pipeline components
from rag_pipeline import (
    BGEEmbedder,
    SingleStoreVectorDB,
    PDFExtractor,
    ChunkingStrategy
)
from advanced_rag import (
    AdvancedRAGPipeline,
    Reranker,
    HybridRetriever,
    create_production_pipeline
)


def example_1_basic_pipeline():
    """
    Example 1: Basic RAG Pipeline
    Fast setup for simple use cases
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic RAG Pipeline")
    print("="*60 + "\n")
    
    # Database configuration
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Initialize components
    print("Loading BGE-M3 model...")
    embedder = BGEEmbedder("BAAI/bge-m3")
    
    print("Connecting to SingleStore...")
    vector_db = SingleStoreVectorDB(**config)
    
    print("Creating embeddings table...")
    vector_db.create_table("basic_embeddings", embedder.embedding_dim)
    
    # Ingest a sample document (you'll need to provide a PDF)
    pdf_path = "sample_document.pdf"
    if Path(pdf_path).exists():
        print(f"\nIngesting {pdf_path}...")
        
        # Extract text
        documents = PDFExtractor.extract_text(pdf_path)
        print(f"Extracted {len(documents)} pages")
        
        # Chunk documents
        chunked_docs = ChunkingStrategy.chunk_documents(
            documents,
            chunk_size=512,
            overlap=128
        )
        print(f"Created {len(chunked_docs)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [doc.content for doc in chunked_docs]
        embeddings = embedder.embed_batch(texts, batch_size=16)
        
        # Store in database
        print("Storing in database...")
        vector_db.insert_embeddings("basic_embeddings", chunked_docs, embeddings)
        print("✓ Ingestion complete!")
        
        # Query the system
        print("\n" + "-"*60)
        print("Querying the system...")
        print("-"*60)
        
        query = "What is the main topic of this document?"
        print(f"Query: {query}\n")
        
        query_embedding = embedder.embed_text(query)
        results = vector_db.search(
            "basic_embeddings",
            query_embedding,
            top_k=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Similarity Score: {result['similarity_score']:.4f}")
            print(f"  Source: {result['source']}")
            print(f"  Page: {result['page_num']}")
            print(f"  Content Preview: {result['content'][:150]}...")
    else:
        print(f"Sample PDF not found at {pdf_path}")
        print("Please provide a PDF file to test ingestion.")


def example_2_advanced_pipeline():
    """
    Example 2: Advanced Pipeline with Hybrid Search and Reranking
    Maximum accuracy for production use
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Pipeline (Maximum Accuracy)")
    print("="*60 + "\n")
    
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Create production pipeline with all features
    print("Initializing advanced pipeline...")
    print("  - Loading BGE-M3 embedder")
    print("  - Loading BGE-Reranker-v2-M3")
    print("  - Enabling hybrid search")
    
    pipeline = create_production_pipeline(
        config,
        use_reranker=True,
        use_hybrid=True
    )
    
    print("✓ Pipeline ready!\n")
    
    # Demonstrate querying with different configurations
    queries = [
        "What are the key findings?",
        "Explain the methodology",
        "What are the conclusions?"
    ]
    
    for query in queries:
        print("-"*60)
        print(f"Query: {query}")
        print("-"*60)
        
        # Query with all accuracy features
        results = pipeline.query(
            query,
            top_k=3,
            retrieval_k=10,  # Retrieve more for reranking
            use_reranking=True
        )
        
        if results:
            print(f"\nFound {len(results)} relevant results:\n")
            
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                
                # Show all available scores
                if 'rerank_score' in result:
                    print(f"  Rerank Score: {result['rerank_score']:.4f}")
                if 'hybrid_score' in result:
                    print(f"  Hybrid Score: {result['hybrid_score']:.4f}")
                if 'dense_score' in result:
                    print(f"  Dense Score: {result['dense_score']:.4f}")
                if 'bm25_score' in result:
                    print(f"  BM25 Score: {result['bm25_score']:.4f}")
                
                print(f"  Source: {result['source']}, Page {result['page_num']}")
                print(f"  Content: {result['content'][:200]}...")
                print()
        else:
            print("No results found.\n")


def example_3_context_retrieval():
    """
    Example 3: Retrieval with Context Windows
    Get surrounding chunks for better LLM context
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Context-Aware Retrieval")
    print("="*60 + "\n")
    
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    pipeline = create_production_pipeline(config)
    
    query = "What is the main argument?"
    print(f"Query: {query}\n")
    
    # Get results with surrounding context
    results, formatted_context = pipeline.query_with_context(
        query,
        top_k=2,
        context_window=1  # Include 1 chunk before and after
    )
    
    print("Results with expanded context:\n")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Main chunk from: {result['source']}, Page {result['page_num']}")
        print(f"  Score: {result.get('rerank_score', result.get('similarity_score')):.4f}")
        
        if 'context_chunks' in result:
            print(f"  Context chunks: {len(result['context_chunks'])}")
            print(f"  Chunk indices: {[c['chunk_index'] for c in result['context_chunks']]}")
    
    print("\n" + "-"*60)
    print("Formatted Context for LLM:")
    print("-"*60)
    print(formatted_context[:500] + "...\n")


def example_4_batch_ingestion():
    """
    Example 4: Batch Ingestion of Multiple PDFs
    Efficient processing of document collections
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Ingestion")
    print("="*60 + "\n")
    
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Initialize
    embedder = BGEEmbedder("BAAI/bge-m3")
    vector_db = SingleStoreVectorDB(**config)
    vector_db.create_table("batch_embeddings", embedder.embedding_dim)
    
    # Process directory of PDFs
    pdf_directory = "./documents"  # Change to your directory
    
    if Path(pdf_directory).exists():
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files\n")
        
        successful = 0
        failed = 0
        
        for pdf_path in pdf_files:
            try:
                print(f"Processing {pdf_path.name}...")
                
                # Extract and chunk
                documents = PDFExtractor.extract_text(str(pdf_path))
                chunked = ChunkingStrategy.chunk_documents(documents)
                
                # Embed and store
                texts = [doc.content for doc in chunked]
                embeddings = embedder.embed_batch(texts, batch_size=16)
                vector_db.insert_embeddings("batch_embeddings", chunked, embeddings)
                
                print(f"  ✓ Processed {len(chunked)} chunks")
                successful += 1
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Batch ingestion complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"{'='*60}\n")
    else:
        print(f"Directory {pdf_directory} not found.")
        print("Create the directory and add PDF files to test batch ingestion.")


def example_5_parameter_comparison():
    """
    Example 5: Compare Different Retrieval Strategies
    Understand tradeoffs between speed and accuracy
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Retrieval Strategy Comparison")
    print("="*60 + "\n")
    
    import time
    
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    query = "What are the main conclusions?"
    
    strategies = [
        ("Dense Only (Fast)", False, False),
        ("Hybrid Search", True, False),
        ("Hybrid + Reranking (Best)", True, True)
    ]
    
    for name, use_hybrid, use_reranker in strategies:
        print(f"\n{name}:")
        print("-" * 40)
        
        pipeline = create_production_pipeline(
            config,
            use_reranker=use_reranker,
            use_hybrid=use_hybrid
        )
        
        start = time.time()
        results = pipeline.query(
            query,
            top_k=5,
            retrieval_k=20,
            use_reranking=use_reranker
        )
        elapsed = time.time() - start
        
        print(f"Time: {elapsed*1000:.1f}ms")
        print(f"Results: {len(results)}")
        
        if results:
            top_result = results[0]
            score_key = 'rerank_score' if use_reranker else 'hybrid_score' if use_hybrid else 'similarity_score'
            print(f"Top Score: {top_result.get(score_key, 0):.4f}")
            print(f"Content: {top_result['content'][:100]}...")


def main():
    """Run all examples"""
    
    print("\n" + "="*60)
    print("HIGH-ACCURACY RAG PIPELINE EXAMPLES")
    print("="*60)
    
    examples = [
        ("1", "Basic Pipeline", example_1_basic_pipeline),
        ("2", "Advanced Pipeline (Max Accuracy)", example_2_advanced_pipeline),
        ("3", "Context-Aware Retrieval", example_3_context_retrieval),
        ("4", "Batch Ingestion", example_4_batch_ingestion),
        ("5", "Strategy Comparison", example_5_parameter_comparison)
    ]
    
    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print("  0. Run all examples")
    
    choice = input("\nSelect example to run (0-5): ").strip()
    
    if choice == "0":
        for _, _, func in examples:
            try:
                func()
            except Exception as e:
                print(f"Error running example: {e}")
                continue
    else:
        for num, name, func in examples:
            if choice == num:
                try:
                    func()
                except Exception as e:
                    print(f"Error: {e}")
                    print("\nMake sure:")
                    print("1. SingleStore is running and configured")
                    print("2. Environment variables are set")
                    print("3. Required packages are installed")
                break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    # Check dependencies
    try:
        import pymysql
        import sentence_transformers
        import PyPDF2
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    main()
