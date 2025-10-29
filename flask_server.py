"""
Flask RAG Server - Refactored with Modular Architecture
Provides REST API for document ingestion and querying
Uses separate modules for S3 and document processing
"""

import os
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from advanced_rag import create_production_pipeline, AdvancedRAGPipeline
from s3_manager import get_s3_config_from_env, create_s3_manager_from_config, S3Manager
from document_processor import DocumentProcessor, ProcessingJobTracker


# ============================================
# Configuration
# ============================================

# Centralized table name
CENTRAL_TABLE = "document_embeddings"

# Upload directory (temporary storage)
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
    'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
    'user': os.getenv('SINGLESTORE_USER', 'root'),
    'password': os.getenv('SINGLESTORE_PASSWORD', ''),
    'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
}

# S3 Configuration
S3_CONFIG = get_s3_config_from_env()

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Job tracking
job_tracker = ProcessingJobTracker()
job_lock = threading.Lock()


# ============================================
# Global Instances
# ============================================

pipeline: Optional[AdvancedRAGPipeline] = None
s3_manager: Optional[S3Manager] = None
document_processor: Optional[DocumentProcessor] = None


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


def get_s3_manager() -> Optional[S3Manager]:
    """Get or create S3 manager instance"""
    global s3_manager
    if s3_manager is None:
        s3_manager = create_s3_manager_from_config(S3_CONFIG)
    return s3_manager


def get_document_processor() -> DocumentProcessor:
    """Get or create document processor instance"""
    global document_processor
    if document_processor is None:
        rag = get_pipeline()
        s3 = get_s3_manager()
        
        document_processor = DocumentProcessor(
            rag_pipeline=rag,
            s3_manager=s3,
            table_name=CENTRAL_TABLE,
            auto_cleanup=True
        )
    return document_processor


# ============================================
# Background Processing Functions
# ============================================

def process_document_async(
    file_path: Path,
    filename: str,
    metadata: Dict[str, Any],
    job_id: str,
    upload_to_s3: bool = False
):
    """Process document in background thread"""
    try:
        processor = get_document_processor()
        
        with job_lock:
            job_tracker.update_job(
                job_id,
                status='processing',
                current_file=filename
            )
        
        result = processor.process_document(
            file_path,
            filename,
            metadata,
            upload_to_s3=upload_to_s3,
            job_id=job_id
        )
        
        with job_lock:
            job_tracker.update_job(job_id, result=result)
        
    except Exception as e:
        error_result = {
            'filename': filename,
            'status': 'failed',
            'error': str(e)
        }
        
        with job_lock:
            job_tracker.update_job(job_id, result=error_result)


# ============================================
# Flask Application
# ============================================

app = Flask(__name__)
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# Health & Info Endpoints
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "RAG API Server (Flask - Refactored)",
        "version": "2.0.0",
        "s3_enabled": S3_CONFIG['enabled'],
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "upload": "/upload (POST)",
            "batch_upload": "/upload/batch (POST)",
            "query": "/query (POST)",
            "search": "/search (GET)",
            "documents": "/documents (GET)",
            "delete": "/documents/<source> (DELETE)",
            "download": "/documents/<source>/download (GET)",
            "jobs": "/jobs (GET)",
            "job_status": "/jobs/<job_id> (GET)"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        rag = get_pipeline()
        
        # Get table statistics
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(DISTINCT source) FROM {CENTRAL_TABLE}")
        doc_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        cursor.execute(f"SELECT COUNT(*) FROM {CENTRAL_TABLE}")
        chunk_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database": DB_CONFIG['database'],
            "table": CENTRAL_TABLE,
            "documents": doc_count,
            "chunks": chunk_count,
            "s3_enabled": S3_CONFIG['enabled']
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get detailed statistics"""
    try:
        rag = get_pipeline()
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        # Total documents and chunks
        cursor.execute(f"SELECT COUNT(DISTINCT source) FROM {CENTRAL_TABLE}")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM {CENTRAL_TABLE}")
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
        
        return jsonify({
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "by_type": by_type,
            "by_department": by_dept,
            "recent_ingestions": recent
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Query Endpoints
# ============================================

@app.route('/query', methods=['POST'])
def query_documents():
    """Query the RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 5)
        retrieval_k = data.get('retrieval_k', 20)
        use_reranking = data.get('use_reranking', True)
        min_score = data.get('min_score')
        
        rag = get_pipeline()
        start_time = time.time()
        
        results = rag.query(
            query_text,
            top_k=top_k,
            retrieval_k=retrieval_k,
            use_reranking=use_reranking,
            score_threshold=min_score
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "query": query_text,
            "results": results,
            "total_results": len(results),
            "retrieval_time_ms": retrieval_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['GET'])
def simple_search():
    """Simple search endpoint (GET request)"""
    try:
        query = request.args.get('q')
        if not query:
            return jsonify({"error": "Missing 'q' parameter"}), 400
        
        top_k = int(request.args.get('top_k', 5))
        
        rag = get_pipeline()
        start_time = time.time()
        
        results = rag.query(
            query,
            top_k=top_k,
            retrieval_k=20,
            use_reranking=True
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "query": query,
            "results": results,
            "total_results": len(results),
            "retrieval_time_ms": retrieval_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Upload Endpoints
# ============================================

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload a single document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        # Get form data
        document_type = request.form.get('document_type', 'document')
        department = request.form.get('department', 'general')
        tags = request.form.get('tags', '')
        author = request.form.get('author')
        project_id = request.form.get('project_id')
        client_name = request.form.get('client_name')
        upload_to_s3 = request.form.get('upload_to_s3', str(S3_CONFIG['enabled'])).lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = UPLOAD_DIR / filename
        file.save(str(file_path))
        
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
        processor = get_document_processor()
        result = processor.process_document(
            file_path,
            filename,
            metadata,
            upload_to_s3=upload_to_s3
        )
        
        # Generate presigned URL if file is in S3
        presigned_url = None
        if result.get('s3_key') and upload_to_s3:
            s3 = get_s3_manager()
            if s3:
                try:
                    presigned_url = s3.generate_presigned_url(result['s3_key'])
                except Exception as e:
                    print(f"Failed to generate presigned URL: {e}")
        
        return jsonify({
            "filename": filename,
            "status": "success",
            "chunks_created": result['chunks'],
            "pages": result['pages'],
            "message": f"Document ingested successfully: {result['chunks']} chunks from {result['pages']} pages",
            "s3_uploaded": bool(result.get('s3_uri')),
            "s3_uri": result.get('s3_uri'),
            "download_url": presigned_url,
            "local_file_deleted": result.get('local_file_deleted', False)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload/batch', methods=['POST'])
def upload_batch():
    """Upload multiple documents at once"""
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        # Get form data
        document_type = request.form.get('document_type', 'document')
        department = request.form.get('department', 'general')
        tags = request.form.get('tags', '')
        author = request.form.get('author')
        project_id = request.form.get('project_id')
        client_name = request.form.get('client_name')
        async_processing = request.form.get('async_processing', 'false').lower() == 'true'
        upload_to_s3 = request.form.get('upload_to_s3', str(S3_CONFIG['enabled'])).lower() == 'true'
        
        # Create job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(files)}"
        
        # Initialize job tracking
        with job_lock:
            job_tracker.create_job(job_id, len(files))
        
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
        
        # Validate and save all files
        file_paths = []
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({
                    "error": f"File {file.filename} is not a PDF"
                }), 400
            
            filename = secure_filename(file.filename)
            file_path = UPLOAD_DIR / filename
            file.save(str(file_path))
            file_paths.append((file_path, filename))
        
        if not file_paths:
            return jsonify({"error": "No valid PDF files found"}), 400
        
        if async_processing:
            # Process in background
            for file_path, filename in file_paths:
                executor.submit(
                    process_document_async,
                    file_path,
                    filename,
                    metadata,
                    job_id,
                    upload_to_s3
                )
            
            return jsonify({
                "total_files": len(files),
                "successful": 0,
                "failed": 0,
                "job_id": job_id,
                "status": "queued",
                "message": f"Batch processing started. Check /jobs/{job_id} for status"
            })
        else:
            # Process synchronously
            processor = get_document_processor()
            batch_result = processor.process_batch(
                file_paths,
                metadata,
                upload_to_s3=upload_to_s3,
                job_id=job_id
            )
            
            # Add presigned URLs if S3 enabled
            s3 = get_s3_manager()
            if s3 and upload_to_s3:
                for file_result in batch_result['files']:
                    if file_result.get('s3_key'):
                        try:
                            file_result['download_url'] = s3.generate_presigned_url(
                                file_result['s3_key']
                            )
                        except Exception as e:
                            print(f"Failed to generate presigned URL: {e}")
            
            return jsonify({
                "total_files": batch_result['total'],
                "successful": batch_result['successful'],
                "failed": batch_result['failed'],
                "job_id": job_id,
                "files": batch_result['files']
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Job Management Endpoints
# ============================================

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get status of a batch processing job"""
    with job_lock:
        job = job_tracker.get_job(job_id)
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        return jsonify(job)


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    with job_lock:
        return jsonify({
            "jobs": job_tracker.get_all_jobs()
        })


# ============================================
# Document Management Endpoints
# ============================================

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents with optional filtering"""
    try:
        document_type = request.args.get('document_type')
        department = request.args.get('department')
        limit = int(request.args.get('limit', 100))
        
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
        
        return jsonify({
            "documents": documents,
            "total": len(documents)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents/<source>', methods=['DELETE'])
def delete_document(source: str):
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
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({
            "message": f"Deleted {deleted_count} chunks from {source}",
            "deleted_chunks": deleted_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents/<source>/download', methods=['GET'])
def download_document(source: str):
    """Get download URL for a document (if in S3)"""
    try:
        # This endpoint would need S3 URI stored in metadata
        # For now, return not implemented
        return jsonify({
            "error": "Download endpoint requires S3 URI in metadata",
            "message": "This feature is available when documents are uploaded to S3"
        }), 501
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Initialize and Run
# ============================================

def initialize():
    """Initialize all components"""
    try:
        # Initialize pipeline
        print("\nInitializing RAG pipeline...")
        rag = get_pipeline()
        print(f"✓ RAG Pipeline initialized")
        print(f"✓ Using centralized table: {CENTRAL_TABLE}")
        
        # Initialize S3 if enabled
        if S3_CONFIG['enabled']:
            print("\nInitializing S3...")
            s3 = get_s3_manager()
            if s3:
                print(f"✓ S3 integration enabled")
            else:
                print(f"⚠️  S3 enabled but initialization failed")
        else:
            print(f"\nℹ️  S3 integration disabled")
        
        # Initialize document processor
        print("\nInitializing document processor...")
        processor = get_document_processor()
        print(f"✓ Document processor ready")
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG API Server (Flask - Refactored)")
    print("=" * 60)
    print(f"Centralized Table: {CENTRAL_TABLE}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"S3 Enabled: {S3_CONFIG['enabled']}")
    if S3_CONFIG['enabled']:
        print(f"S3 Bucket: {S3_CONFIG['bucket']}")
    print("=" * 60)
    print()
    
    # Initialize before starting server
    print("Initializing components...")
    initialize()
    print()
    
    print("=" * 60)
    print("Server ready!")
    print("API Documentation: http://localhost:8000/")
    print("=" * 60)
    print()
    
    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        threaded=True
    )
