"""
FastAPI RAG Server with Centralized Table and Batch Upload
Provides REST API for document ingestion and querying
"""

import os
import io
import json
import shutil
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from advanced_rag import create_production_pipeline, AdvancedRAGPipeline
from rag_pipeline import PDFExtractor, ChunkingStrategy


# ============================================
# Configuration
# ============================================

# Centralized table name - all documents go here
CENTRAL_TABLE = "document_embeddings"

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Temporary processing directory
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
    'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
    'user': os.getenv('SINGLESTORE_USER', 'root'),
    'password': os.getenv('SINGLESTORE_PASSWORD', ''),
    'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
}

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)


# ============================================
# Pydantic Models
# ============================================

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    retrieval_k: int = Field(20, ge=5, le=100, description="Candidates for reranking")
    use_reranking: bool = Field(True, description="Enable reranking")
    document_type: Optional[str] = Field(None, description="Filter by document type")
    department: Optional[str] = Field(None, description="Filter by department")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum relevance score")


class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    retrieval_time_ms: float


class UploadResponse(BaseModel):
    """Response model for uploads"""
    filename: str
    status: str
    chunks_created: Optional[int] = None
    pages: Optional[int] = None
    message: Optional[str] = None
    job_id: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """Response model for batch uploads"""
    total_files: int
    successful: int
    failed: int
    job_id: str
    files: List[UploadResponse]


class DocumentMetadata(BaseModel):
    """Metadata for document uploads"""
    document_type: str = Field("document", description="Type of document")
    department: str = Field("general", description="Department")
    tags: List[str] = Field(default_factory=list, description="Tags")
    author: Optional[str] = Field(None, description="Author")
    date: Optional[str] = Field(None, description="Document date")
    project_id: Optional[str] = Field(None, description="Project ID")
    client_name: Optional[str] = Field(None, description="Client name")
    security_level: str = Field("internal", description="Security level")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database: str
    table: str
    documents: int
    chunks: int


# ============================================
# Global Pipeline Instance
# ============================================

pipeline: Optional[AdvancedRAGPipeline] = None


def get_pipeline() -> AdvancedRAGPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = create_production_pipeline(
            DB_CONFIG,
            use_reranker=True,
            use_hybrid=True,
            table_name=CENTRAL_TABLE
        )
    return pipeline


# ============================================
# Background Processing
# ============================================

# Job tracking
processing_jobs: Dict[str, Dict[str, Any]] = {}


async def process_document_async(
    file_path: Path,
    filename: str,
    metadata: Dict[str, Any],
    job_id: str
):
    """Process a document in the background"""
    try:
        # Update job status
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['current_file'] = filename
        
        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            process_document_sync,
            file_path,
            filename,
            metadata
        )
        
        # Update job
        processing_jobs[job_id]['processed_files'].append({
            'filename': filename,
            'status': 'success',
            'chunks': result['chunks'],
            'pages': result['pages']
        })
        processing_jobs[job_id]['successful'] += 1
        
        return result
        
    except Exception as e:
        processing_jobs[job_id]['processed_files'].append({
            'filename': filename,
            'status': 'failed',
            'error': str(e)
        })
        processing_jobs[job_id]['failed'] += 1
        raise


def process_document_sync(
    file_path: Path,
    filename: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Synchronous document processing (runs in thread pool)"""
    try:
        rag = get_pipeline()
        
        # Extract text from PDF
        documents = PDFExtractor.extract_text(str(file_path))
        
        if not documents:
            raise ValueError("No text could be extracted from PDF")
        
        # Chunk documents
        chunked_docs = ChunkingStrategy.chunk_documents(documents)
        
        # Add metadata to all chunks
        file_metadata = {
            "document_id": file_path.stem,
            "source_file": filename,
            "file_path": str(file_path),
            "total_pages": len(documents),
            "total_chunks": len(chunked_docs),
            "file_size_kb": file_path.stat().st_size // 1024,
            "ingestion_date": datetime.now().isoformat(),
            **metadata
        }
        
        for chunk in chunked_docs:
            chunk.metadata = {**chunk.metadata, **file_metadata}
        
        # Generate embeddings
        texts = [doc.content for doc in chunked_docs]
        embeddings = rag.embedder.embed_batch(texts)
        
        # Store in centralized table
        rag.vector_db.insert_embeddings(
            CENTRAL_TABLE,
            chunked_docs,
            embeddings
        )
        
        return {
            'filename': filename,
            'chunks': len(chunked_docs),
            'pages': len(documents)
        }
        
    except Exception as e:
        raise Exception(f"Failed to process {filename}: {str(e)}")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="RAG API Server",
    description="Centralized RAG system with document upload and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        # Initialize pipeline and ensure table exists
        rag = get_pipeline()
        print(f"âœ“ RAG Pipeline initialized")
        print(f"âœ“ Using centralized table: {CENTRAL_TABLE}")
    except Exception as e:
        print(f"âœ— Failed to initialize: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    executor.shutdown(wait=True)


# ============================================
# Health & Info Endpoints
# ============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_pipeline()
        
        # Get table statistics
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        # Count documents and chunks
        cursor.execute(f"SELECT COUNT(DISTINCT source) FROM {CENTRAL_TABLE}")
        doc_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        cursor.execute(f"SELECT COUNT(*) FROM {CENTRAL_TABLE}")
        chunk_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        cursor.close()
        conn.close()
        
        return HealthResponse(
            status="healthy",
            database=DB_CONFIG['database'],
            table=CENTRAL_TABLE,
            documents=doc_count,
            chunks=chunk_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get detailed statistics"""
    try:
        rag = get_pipeline()
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        # Total documents and chunks
        cursor.execute(f"SELECT COUNT(DISTINCT source) as doc_count FROM {CENTRAL_TABLE}")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) as chunk_count FROM {CENTRAL_TABLE}")
        chunk_count = cursor.fetchone()[0]
        
        # By document type
        cursor.execute(f"""
            SELECT 
                JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.document_type')) as type,
                COUNT(DISTINCT source) as count
            FROM {CENTRAL_TABLE}
            GROUP BY type
            ORDER BY count DESC
        """)
        by_type = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # By department
        cursor.execute(f"""
            SELECT 
                JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.department')) as dept,
                COUNT(DISTINCT source) as count
            FROM {CENTRAL_TABLE}
            GROUP BY dept
            ORDER BY count DESC
        """)
        by_dept = [{"department": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Recent ingestions
        cursor.execute(f"""
            SELECT 
                JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.source_file')) as file,
                JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.ingestion_date')) as date,
                COUNT(*) as chunks
            FROM {CENTRAL_TABLE}
            GROUP BY file, date
            ORDER BY date DESC
            LIMIT 10
        """)
        recent = [
            {"filename": row[0], "date": row[1], "chunks": row[2]}
            for row in cursor.fetchall()
        ]
        
        cursor.close()
        conn.close()
        
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "by_type": by_type,
            "by_department": by_dept,
            "recent_ingestions": recent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Query Endpoints
# ============================================

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        import time
        rag = get_pipeline()
        
        start_time = time.time()
        
        # Query with filters if provided
        if request.document_type or request.department:
            # Custom query with metadata filtering
            conn = rag.vector_db.get_connection()
            cursor = conn.cursor()
            
            query_embedding = rag.embedder.embed_text(request.query)
            query_json = json.dumps(query_embedding.tolist())
            
            where_clauses = []
            params = [query_json]
            
            if request.document_type:
                where_clauses.append("JSON_EXTRACT(metadata, '$.document_type') = %s")
                params.append(request.document_type)
            
            if request.department:
                where_clauses.append("JSON_EXTRACT(metadata, '$.department') = %s")
                params.append(request.department)
            
            if request.min_score:
                where_clauses.append("DOT_PRODUCT(embedding, JSON_ARRAY_PACK(%s)) >= %s")
                params.append(request.min_score)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            params.append(request.top_k)
            
            sql = f"""
            SELECT 
                content, source, page_num, chunk_index, metadata,
                DOT_PRODUCT(embedding, JSON_ARRAY_PACK(%s)) as score
            FROM {CENTRAL_TABLE}
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT %s
            """
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "content": row[0],
                    "source": row[1],
                    "page_num": row[2],
                    "chunk_index": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "similarity_score": float(row[5])
                })
            
            cursor.close()
            conn.close()
            
        else:
            # Standard query
            results = rag.query(
                request.query,
                top_k=request.top_k,
                retrieval_k=request.retrieval_k,
                use_reranking=request.use_reranking,
                score_threshold=request.min_score
            )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            retrieval_time_ms=retrieval_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def simple_search(
    q: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
    department: Optional[str] = None
):
    """Simple search endpoint (GET request)"""
    request = QueryRequest(
        query=q,
        top_k=top_k,
        document_type=document_type,
        department=department
    )
    return await query_documents(request)


# ============================================
# Upload Endpoints
# ============================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("document"),
    department: str = Form("general"),
    tags: str = Form(""),
    author: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None)
):
    """
    Upload a single document
    
    - **file**: PDF file to upload
    - **document_type**: Type of document (contract, report, manual, etc.)
    - **department**: Department (legal, engineering, sales, etc.)
    - **tags**: Comma-separated tags
    - **author**: Document author
    - **project_id**: Related project ID
    - **client_name**: Related client name
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare metadata
        metadata = {
            "document_type": document_type,
            "department": department,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "author": author,
            "project_id": project_id,
            "client_name": client_name,
            "security_level": "internal"
        }
        
        # Process document
        result = process_document_sync(file_path, file.filename, metadata)
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            chunks_created=result['chunks'],
            pages=result['pages'],
            message=f"Document ingested successfully: {result['chunks']} chunks from {result['pages']} pages"
        )
        
    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_type: str = Form("document"),
    department: str = Form("general"),
    tags: str = Form(""),
    author: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
    async_processing: bool = Form(False)
):
    """
    Upload multiple documents at once
    
    - **files**: List of PDF files
    - **async_processing**: If True, process in background and return job ID
    - Other params same as single upload
    """
    try:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(files)}"
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            'job_id': job_id,
            'total_files': len(files),
            'successful': 0,
            'failed': 0,
            'status': 'pending',
            'processed_files': [],
            'started_at': datetime.now().isoformat()
        }
        
        # Prepare metadata
        metadata = {
            "document_type": document_type,
            "department": department,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "author": author,
            "project_id": project_id,
            "client_name": client_name,
            "security_level": "internal"
        }
        
        results = []
        
        # Validate all files first
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
        
        # Save all files
        file_paths = []
        for file in files:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append((file_path, file.filename))
        
        if async_processing:
            # Process in background
            for file_path, filename in file_paths:
                background_tasks.add_task(
                    process_document_async,
                    file_path,
                    filename,
                    metadata,
                    job_id
                )
            
            return BatchUploadResponse(
                total_files=len(files),
                successful=0,
                failed=0,
                job_id=job_id,
                files=[
                    UploadResponse(
                        filename=filename,
                        status="queued",
                        job_id=job_id
                    )
                    for _, filename in file_paths
                ]
            )
        else:
            # Process synchronously
            for file_path, filename in file_paths:
                try:
                    result = process_document_sync(file_path, filename, metadata)
                    results.append(UploadResponse(
                        filename=filename,
                        status="success",
                        chunks_created=result['chunks'],
                        pages=result['pages'],
                        message=f"Processed successfully"
                    ))
                    processing_jobs[job_id]['successful'] += 1
                except Exception as e:
                    results.append(UploadResponse(
                        filename=filename,
                        status="failed",
                        message=str(e)
                    ))
                    processing_jobs[job_id]['failed'] += 1
            
            processing_jobs[job_id]['status'] = 'completed'
            
            return BatchUploadResponse(
                total_files=len(files),
                successful=processing_jobs[job_id]['successful'],
                failed=processing_jobs[job_id]['failed'],
                job_id=job_id,
                files=results
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a batch processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "jobs": list(processing_jobs.values())
    }


# ============================================
# Document Management Endpoints
# ============================================

@app.get("/documents")
async def list_documents(
    document_type: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 100
):
    """List all documents with optional filtering"""
    try:
        rag = get_pipeline()
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        where_clauses = []
        params = []
        
        if document_type:
            where_clauses.append("JSON_EXTRACT(metadata, '$.document_type') = %s")
            params.append(document_type)
        
        if department:
            where_clauses.append("JSON_EXTRACT(metadata, '$.department') = %s")
            params.append(department)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(limit)
        
        sql = f"""
        SELECT DISTINCT
            source,
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.document_id')) as doc_id,
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.document_type')) as type,
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.department')) as dept,
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.total_pages')) as pages,
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.ingestion_date')) as ingested,
            COUNT(*) as chunks
        FROM {CENTRAL_TABLE}
        WHERE {where_sql}
        GROUP BY source, doc_id, type, dept, pages, ingested
        ORDER BY ingested DESC
        LIMIT %s
        """
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            documents.append({
                "source": row[0],
                "document_id": row[1],
                "type": row[2],
                "department": row[3],
                "pages": row[4],
                "ingestion_date": row[5],
                "chunks": row[6]
            })
        
        cursor.close()
        conn.close()
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{source}")
async def delete_document(source: str):
    """Delete a document by source filename"""
    try:
        rag = get_pipeline()
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        # Delete all chunks from this document
        cursor.execute(f"DELETE FROM {CENTRAL_TABLE} WHERE source = %s", (source,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "message": f"Deleted {deleted_count} chunks from {source}",
            "deleted_chunks": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("RAG API Server")
    print("=" * 60)
    print(f"Centralized Table: {CENTRAL_TABLE}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Database: {DB_CONFIG['database']}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
