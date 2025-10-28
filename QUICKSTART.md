# Quick Start Guide

Get your RAG pipeline running in 5 minutes!

## Prerequisites

1. **SingleStore 8.9+** installed and running
2. **Python 3.8+** installed
3. **4GB+ RAM** available
4. **Internet connection** for first-time model download

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- sentence-transformers (for BGE-M3)
- pymysql (for SingleStore)
- PyPDF2 (for PDF extraction)
- numpy (for array operations)

### 2. Configure Database

Create a `.env` file:

```bash
cp .env.template .env
```

Edit `.env` with your SingleStore credentials:

```bash
SINGLESTORE_HOST=localhost
SINGLESTORE_PORT=3306
SINGLESTORE_USER=your_user
SINGLESTORE_PASSWORD=your_password
SINGLESTORE_DATABASE=rag_db
```

### 3. Create Database

```sql
CREATE DATABASE rag_db;
```

The pipeline will automatically create the required tables.

### 4. Test Your Setup

```bash
python test_pipeline.py
```

This will verify:
- Database connection
- Model loading
- PDF extraction
- Embedding storage

## Your First RAG Query (5 lines of code!)

```python
from advanced_rag import create_production_pipeline
import os

# Load config from environment
config = {
    'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
    'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
    'user': os.getenv('SINGLESTORE_USER', 'root'),
    'password': os.getenv('SINGLESTORE_PASSWORD', ''),
    'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
}

# Create pipeline (loads models automatically)
pipeline = create_production_pipeline(config)

# Ingest a PDF (this will take a minute on first run)
from rag_pipeline import PDFExtractor, ChunkingStrategy
docs = PDFExtractor.extract_text("your_document.pdf")
chunks = ChunkingStrategy.chunk_documents(docs)
texts = [c.content for c in chunks]
embeddings = pipeline.embedder.embed_batch(texts)
pipeline.vector_db.insert_embeddings("document_embeddings", chunks, embeddings)

# Query!
results = pipeline.query("What is this document about?", top_k=5)

# Print results
for i, r in enumerate(results, 1):
    print(f"\n{i}. Score: {r.get('rerank_score', r.get('similarity_score')):.3f}")
    print(f"   {r['content'][:200]}...")
```

## Common Issues & Solutions

### "Connection refused" Error
**Problem**: Can't connect to SingleStore  
**Solution**: 
```bash
# Check if SingleStore is running
systemctl status singlestoredb  # Linux
# or
brew services list  # Mac

# Verify port is open
telnet localhost 3306
```

### "Model not found" Error
**Problem**: First-time model download failing  
**Solution**:
```bash
# Ensure internet connection
# BGE-M3 is ~2GB, may take time
# Check disk space: need ~5GB free
df -h
```

### "Out of memory" Error
**Problem**: Not enough RAM for embeddings  
**Solution**:
```python
# Reduce batch size
embeddings = embedder.embed_batch(texts, batch_size=8)  # default is 32

# Or process in smaller chunks
for i in range(0, len(texts), 100):
    batch = texts[i:i+100]
    batch_emb = embedder.embed_batch(batch, batch_size=8)
    # store batch_emb
```

### Slow Queries
**Problem**: Queries taking >2 seconds  
**Solution**:
```python
# Option 1: Disable reranking for speed
results = pipeline.query(query, use_reranking=False)

# Option 2: Use basic pipeline instead
from rag_pipeline import RAGPipeline, BGEEmbedder, SingleStoreVectorDB
embedder = BGEEmbedder("BAAI/bge-m3")
vector_db = SingleStoreVectorDB(**config)
pipeline = RAGPipeline(embedder, vector_db)
```

## What Files Do What?

- **rag_pipeline.py**: Core pipeline with PDF extraction, chunking, embedding, storage
- **advanced_rag.py**: Advanced features (hybrid search, reranking, context)
- **example_usage.py**: Complete working examples
- **test_pipeline.py**: Validation and testing utilities
- **requirements.txt**: Python dependencies
- **README.md**: Full documentation
- **.env.template**: Configuration template

## Next Steps

1. **Run examples**: `python example_usage.py`
2. **Read full docs**: Open `README.md`
3. **Customize**: Adjust chunk size, overlap, alpha in your code
4. **Optimize**: See README.md for tuning guide

## Quick Reference

### Best Defaults
- Chunk size: 512
- Overlap: 128  
- Top-K: 5
- Retrieval-K: 20 (for reranking)
- Hybrid alpha: 0.5

### Performance Targets
- Ingestion: 10-50 pages/min
- Query (basic): ~50-100ms
- Query (advanced): ~500ms

### Storage Requirements
- Per chunk: ~4.5KB (1KB text + 4KB embedding)
- 1000 pages ≈ 4MB
- 100K pages ≈ 400MB

## Getting Help

1. Check `README.md` for detailed documentation
2. Run `python test_pipeline.py` to diagnose issues
3. Review `example_usage.py` for working code patterns

## Pro Tips

- **Always test first**: Run `test_pipeline.py` before production use
- **Start simple**: Use basic pipeline, add features as needed
- **Benchmark**: Use example 5 to compare strategies
- **Monitor scores**: Scores <0.3 may indicate poor retrieval
- **Tune chunk size**: 256 for short docs, 512 for medium, 1024 for long

Happy RAG building! 🚀
