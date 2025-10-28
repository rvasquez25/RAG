# High-Accuracy RAG Pipeline with BGE-M3 and SingleStore

A production-ready RAG (Retrieval-Augmented Generation) pipeline optimized for maximum accuracy with unstructured documents, particularly PDFs.

## Features

### Core Components
- **BGE-M3 Embeddings**: State-of-the-art multilingual dense embeddings (1024 dimensions)
- **SingleStore Vector Database**: High-performance vector storage with native vector operations
- **Semantic Chunking**: Intelligent text chunking that respects document structure
- **Hybrid Search**: Combines dense vector search with BM25 sparse retrieval
- **Cross-Encoder Reranking**: Final reranking step for maximum accuracy

### Accuracy Enhancements
1. **Multi-stage Retrieval**:
   - Initial retrieval: Dense + Sparse (Hybrid)
   - Reranking: Cross-encoder for final results
   - Context expansion: Include surrounding chunks

2. **Smart Chunking**:
   - Respects paragraph boundaries
   - Configurable overlap to prevent information loss
   - Maintains document hierarchy metadata

3. **Configurable Parameters**:
   - Chunk size: 512 tokens (optimal for BGE-M3)
   - Overlap: 128 tokens (25% overlap recommended)
   - Retrieval depth: Top-20 for reranking
   - Final results: Top-5 with highest scores

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: For better PDF extraction
pip install pdfplumber

# Optional: For OCR support (scanned PDFs)
pip install pytesseract
apt-get install tesseract-ocr  # Linux
```

## SingleStore Setup

### 1. Create Database

```sql
CREATE DATABASE rag_db;
USE rag_db;
```

### 2. Verify Vector Support

```sql
-- SingleStore 8.9+ has native vector support
SELECT @@version;
```

### 3. Table Schema (Optimized for SingleStore 8.9+)

The pipeline automatically creates an optimized table schema. The table uses a composite key design to comply with SingleStore 8.9's sharding requirements:

```sql
CREATE TABLE document_embeddings (
    id BIGINT AUTO_INCREMENT,
    chunk_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(512) NOT NULL,
    page_num INT,
    chunk_index INT,
    metadata JSON,
    embedding VECTOR(1024) NOT NULL,  -- BGE-M3 dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, chunk_id),
    SHARD KEY (id),
    UNIQUE KEY (chunk_id, id),
    KEY (source),
    KEY (page_num),
    KEY (source, page_num)
);
```

**Note:** For detailed explanation of the schema design and SingleStore 8.9 compatibility, see [SINGLESTORE_COMPATIBILITY.md](SINGLESTORE_COMPATIBILITY.md).

## Configuration

### Environment Variables

Create a `.env` file:

```bash
SINGLESTORE_HOST=localhost
SINGLESTORE_PORT=3306
SINGLESTORE_USER=your_user
SINGLESTORE_PASSWORD=your_password
SINGLESTORE_DATABASE=rag_db
```

## Usage

### Basic Pipeline (Fast)

```python
from rag_pipeline import RAGPipeline, BGEEmbedder, SingleStoreVectorDB

# Configuration
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'rag_db'
}

# Initialize
embedder = BGEEmbedder("BAAI/bge-m3")
vector_db = SingleStoreVectorDB(**config)
pipeline = RAGPipeline(embedder, vector_db)

# Ingest documents
pipeline.ingest_pdf("document.pdf")
pipeline.ingest_directory("./pdfs")

# Query
results = pipeline.query("What is the main topic?", top_k=5)
for result in results:
    print(f"Score: {result['similarity_score']:.4f}")
    print(f"Source: {result['source']}, Page: {result['page_num']}")
    print(f"Content: {result['content']}\n")
```

### Advanced Pipeline (Maximum Accuracy)

```python
from advanced_rag import create_production_pipeline

# Create advanced pipeline with all accuracy features
pipeline = create_production_pipeline(
    config,
    use_reranker=True,  # Enable cross-encoder reranking
    use_hybrid=True     # Enable hybrid search
)

# Ingest (same as basic)
from rag_pipeline import PDFExtractor, ChunkingStrategy
documents = PDFExtractor.extract_text("document.pdf")
chunked = ChunkingStrategy.chunk_documents(documents, chunk_size=512, overlap=128)

texts = [doc.content for doc in chunked]
embeddings = pipeline.embedder.embed_batch(texts)
pipeline.vector_db.insert_embeddings("document_embeddings", chunked, embeddings)

# Query with all enhancements
results, formatted_context = pipeline.query_with_context(
    "What are the key findings?",
    top_k=5,
    context_window=1  # Include adjacent chunks for context
)

# Use formatted_context with your LLM
print(formatted_context)
```

### Custom Chunking Strategy

```python
from rag_pipeline import ChunkingStrategy

# Adjust parameters based on your documents
chunked_docs = ChunkingStrategy.chunk_documents(
    documents,
    chunk_size=512,   # Larger chunks = more context, fewer chunks
    overlap=128       # More overlap = better continuity, more storage
)
```

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple PDFs efficiently
import os
from pathlib import Path

pdf_dir = "./documents"
batch_size = 10

pdf_files = list(Path(pdf_dir).glob("*.pdf"))
for i in range(0, len(pdf_files), batch_size):
    batch = pdf_files[i:i+batch_size]
    for pdf in batch:
        try:
            pipeline.ingest_pdf(str(pdf))
        except Exception as e:
            print(f"Error with {pdf}: {e}")
```

### 2. Embedding Batch Size

```python
# Adjust batch size based on your GPU memory
embeddings = embedder.embed_batch(
    texts,
    batch_size=32  # Reduce if OOM, increase for faster processing
)
```

### 3. Query Optimization

```python
# For best accuracy/speed tradeoff
results = pipeline.query(
    query_text,
    top_k=5,              # Final results
    retrieval_k=20,       # Initial retrieval (for reranking)
    use_reranking=True,   # Enable for max accuracy
    score_threshold=0.5   # Filter low-quality results
)
```

## Accuracy Tuning Guide

### Retrieval Strategy Selection

| Use Case | Hybrid | Reranker | Retrieval K | Time | Accuracy |
|----------|--------|----------|-------------|------|----------|
| Fast queries | No | No | N/A | ~50ms | Good |
| Balanced | Yes | No | N/A | ~100ms | Better |
| Maximum accuracy | Yes | Yes | 20-50 | ~500ms | Best |
| Domain-specific | Yes | Yes | 50-100 | ~1s | Best |

### Parameter Recommendations

**Chunk Size**:
- Small (256): Better for precise retrieval, more chunks
- Medium (512): **Recommended** - balanced performance
- Large (1024): More context, but may dilute relevance

**Overlap**:
- No overlap (0): Fast, but may lose context at boundaries
- Small (64): Minimal context preservation
- Medium (128): **Recommended** - good balance
- Large (256): Maximum continuity, 2x storage

**Hybrid Alpha** (dense vs sparse weight):
- 0.3-0.4: Better for keyword-heavy queries
- 0.5: **Recommended** - balanced
- 0.6-0.7: Better for semantic queries

**Score Thresholds**:
- Dense similarity: 0.3-0.5
- Hybrid score: 0.4-0.6
- Rerank score: 0.5-0.7

## Advanced Features

### 1. Metadata Filtering

```python
# Add custom metadata during ingestion
doc.metadata = {
    'department': 'engineering',
    'date': '2024-01',
    'category': 'technical'
}

# Query with metadata filter (requires custom SQL)
conn = pymysql.connect(**config)
cursor = conn.cursor()
cursor.execute("""
    SELECT content, DOT_PRODUCT(embedding, %s) as score
    FROM document_embeddings
    WHERE JSON_EXTRACT(metadata, '$.department') = 'engineering'
    ORDER BY score DESC
    LIMIT 5
""", (query_embedding.tolist(),))
```

### 2. Multi-Query Retrieval

```python
# Use multiple query formulations
queries = [
    "What is the main finding?",
    "Key results of the study",
    "Primary conclusions"
]

all_results = []
for q in queries:
    results = pipeline.query(q, top_k=10)
    all_results.extend(results)

# Deduplicate by chunk_id and take top results
seen = set()
unique_results = []
for r in sorted(all_results, key=lambda x: x.get('rerank_score', 0), reverse=True):
    if r['chunk_id'] not in seen:
        seen.add(r['chunk_id'])
        unique_results.append(r)
        if len(unique_results) >= 5:
            break
```

### 3. Document Update Strategy

```python
# The pipeline uses UPSERT logic
# Re-ingesting a document will update existing chunks
pipeline.ingest_pdf("updated_document.pdf")  # Replaces old version

# For deletion, use direct SQL
conn = pymysql.connect(**config)
cursor = conn.cursor()
cursor.execute("DELETE FROM document_embeddings WHERE source = %s", (pdf_path,))
conn.commit()
```

## Monitoring and Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Analyze Search Results

```python
results = pipeline.query(query_text, top_k=10)

print(f"Retrieved {len(results)} results")
print(f"Score range: {results[-1]['similarity_score']:.3f} - {results[0]['similarity_score']:.3f}")

# Check score distribution
scores = [r['similarity_score'] for r in results]
print(f"Mean score: {sum(scores)/len(scores):.3f}")
```

### Performance Profiling

```python
import time

start = time.time()
results = pipeline.query(query_text)
query_time = time.time() - start

print(f"Query completed in {query_time:.2f}s")
print(f"Time per result: {query_time/len(results):.3f}s")
```

## Best Practices

1. **Document Preprocessing**:
   - Remove headers/footers if repetitive
   - Normalize whitespace
   - Consider OCR for scanned PDFs

2. **Chunking Strategy**:
   - Use semantic chunking for narrative documents
   - Use fixed-size for tabular/structured data
   - Maintain metadata about chunk position

3. **Embedding Strategy**:
   - Batch embed for efficiency
   - Normalize embeddings (BGE-M3 does this automatically)
   - Cache embeddings if queries are repeated

4. **Query Strategy**:
   - Use reranking for < 1000 QPS
   - Use hybrid search for diverse document types
   - Adjust retrieval_k based on accuracy needs

5. **Storage Management**:
   - Regular backups of SingleStore database
   - Monitor disk usage (embeddings are 4KB per chunk)
   - Index optimization for large datasets

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Connection refused" to SingleStore
- Check SingleStore is running
- Verify network connectivity
- Check firewall settings

### Low retrieval accuracy
- Increase retrieval_k and enable reranking
- Adjust chunk_size (try 256 or 1024)
- Enable hybrid search
- Check if documents are properly ingested

### Out of memory during embedding
- Reduce batch_size in embed_batch()
- Process documents in smaller batches
- Use CPU instead of GPU for smaller workloads

### Slow query performance
- Disable reranking for speed
- Reduce retrieval_k
- Add database indexes
- Use query caching

## Performance Benchmarks

Typical performance on standard hardware (16GB RAM, 8-core CPU):

- **Ingestion**: ~10-50 pages/minute
- **Embedding**: ~100-500 chunks/second (CPU), ~1000+ chunks/second (GPU)
- **Query (dense only)**: ~50-100ms
- **Query (hybrid)**: ~100-200ms
- **Query (hybrid + rerank)**: ~500-1000ms

## License

This pipeline uses the following models:
- BGE-M3: MIT License
- BGE-Reranker-v2-M3: MIT License

Check the respective model cards on HuggingFace for details.

## Support

For issues specific to:
- BGE models: https://github.com/FlagOpen/FlagEmbedding
- SingleStore: https://docs.singlestore.com/
- This pipeline: Create an issue in your repository
