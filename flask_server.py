"""
Flask RAG Server with Centralized Table, Batch Upload, and S3 Integration
Provides REST API for document ingestion and querying with automatic cleanup
"""

import os
import io
import json
import shutil
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from advanced_rag import create_production_pipeline, AdvancedRAGPipeline
from rag_pipeline import PDFExtractor, ChunkingStrategy

# Optional: Import boto3 for S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("⚠️  boto3 not installed. S3 features disabled.")
    print("   Install with: pip install boto3")


# ============================================
# Configuration
# ============================================

# Centralized table name
CENTRAL_TABLE = "document_embeddings"

# Upload directory (temporary storage)
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

# S3 Configuration (optional)
S3_CONFIG = {
    'bucket': os.getenv('S3_BUCKET', 'my-rag-documents'),
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
    'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'enabled': os.getenv('ENABLE_S3', 'false').lower() == 'true'
}

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Job tracking
processing_jobs: Dict[str, Dict[str, Any]] = {}
job_lock = threading.Lock()


# ============================================
# S3 Helper Class
# ============================================

class S3Manager:
    """Manage S3 uploads and presigned URLs"""
    
    def __init__(self, bucket: str, region: str = 'us-east-1'):
        """Initialize S3 client"""
        if not S3_AVAILABLE:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        self.bucket = bucket
        self.region = region
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=S3_CONFIG['access_key'],
            aws_secret_access_key=S3_CONFIG['secret_key']
        )
        
        # Ensure bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            print(f"✓ Connected to S3 bucket: {bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Creating S3 bucket: {bucket}")
                self.s3_client.create_bucket(Bucket=bucket)
            else:
                raise
    
    def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        metadata: Dict[str, str] = None
    ) -> str:
        """
        Upload file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata dict
            
        Returns:
            S3 URI (s3://bucket/key)
        """
        try:
            extra_args = {}
            if metadata:
                # S3 metadata keys must be lowercase
                extra_args['Metadata'] = {
                    k.lower().replace(' ', '-'): str(v) 
                    for k, v in metadata.items()
                }
            
            self.s3_client.upload_file(
                str(file_path),
                self.bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            print(f"✓ Uploaded to S3: {s3_uri}")
            return s3_uri
            
        except Exception as e:
            print(f"✗ S3 upload failed: {e}")
            raise
    
    def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for downloading
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default 1 hour)
            
        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"✗ Failed to generate presigned URL: {e}")
            raise
    
    def delete_file(self, s3_key: str):
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            print(f"✓ Deleted from S3: {s3_key}")
        except Exception as e:
            print(f"✗ S3 deletion failed: {e}")
            raise


# ============================================
# Global Instances
# ============================================

pipeline: Optional[AdvancedRAGPipeline] = None
s3_manager: Optional[S3Manager] = None


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
    
    if not S3_CONFIG['enabled']:
        return None
    
    if not S3_AVAILABLE:
        print("⚠️  S3 enabled but boto3 not installed")
        return None
    
    if s3_manager is None:
        try:
            s3_manager = S3Manager(
                S3_CONFIG['bucket'],
                S3_CONFIG['region']
            )
        except Exception as e:
            print(f"⚠️  Failed to initialize S3: {e}")
            return None
    
    return s3_manager


# ============================================
# Document Processing Functions
# ============================================

def process_document(
    file_path: Path,
    filename: str,
    metadata: Dict[str, Any],
    job_id: str = None,
    upload_to_s3: bool = False
) -> Dict[str, Any]:
    """
    Process a document: ingest to DB, optionally upload to S3, then delete local file
    
    Args:
        file_path: Local file path
        filename: Original filename
        metadata: Document metadata
        job_id: Optional job ID for tracking
        upload_to_s3: Whether to upload to S3
        
    Returns:
        Result dictionary with processing info
    """
    s3_uri = None
    s3_key = None
    
    try:
        # Update job status
        if job_id:
            with job_lock:
                if job_id in processing_jobs:
                    processing_jobs[job_id]['status'] = 'processing'
                    processing_jobs[job_id]['current_file'] = filename
        
        rag = get_pipeline()
        
        # Extract text from PDF
        print(f"Extracting text from {filename}...")
        documents = PDFExtractor.extract_text(str(file_path))
        
        if not documents:
            raise ValueError("No text could be extracted from PDF")
        
        # Chunk documents
        print(f"Chunking {filename}...")
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
        print(f"Generating embeddings for {filename}...")
        texts = [doc.content for doc in chunked_docs]
        embeddings = rag.embedder.embed_batch(texts)
        
        # Store in centralized table
        print(f"Storing {len(chunked_docs)} chunks in database...")
        rag.vector_db.insert_embeddings(
            CENTRAL_TABLE,
            chunked_docs,
            embeddings
        )
        
        print(f"✓ Successfully ingested {filename} to database")
        
        # Upload to S3 if enabled
        if upload_to_s3:
            s3 = get_s3_manager()
            if s3:
                # Create S3 key: documents/{date}/{filename}
                date_prefix = datetime.now().strftime('%Y/%m/%d')
                s3_key = f"documents/{date_prefix}/{filename}"
                
                print(f"Uploading {filename} to S3...")
                s3_uri = s3.upload_file(
                    file_path,
                    s3_key,
                    metadata={
                        'original-filename': filename,
                        'document-id': file_path.stem,
                        'upload-date': datetime.now().isoformat(),
                        'document-type': metadata.get('document_type', 'document'),
                        'department': metadata.get('department', 'general')
                    }
                )
                
                print(f"✓ Uploaded {filename} to S3: {s3_uri}")
        
        # Delete local file after successful processing
        try:
            file_path.unlink()
            print(f"✓ Deleted local file: {file_path}")
        except Exception as e:
            print(f"⚠️  Failed to delete local file {file_path}: {e}")
            # Don't fail the whole operation if cleanup fails
        
        result = {
            'filename': filename,
            'status': 'success',
            'chunks': len(chunked_docs),
            'pages': len(documents),
            's3_uri': s3_uri,
            's3_key': s3_key,
            'local_file_deleted': True
        }
        
        # Update job tracking
        if job_id:
            with job_lock:
                if job_id in processing_jobs:
                    processing_jobs[job_id]['processed_files'].append(result)
                    processing_jobs[job_id]['successful'] += 1
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed to process {filename}: {error_msg}")
        traceback.print_exc()
        
        # Update job tracking
        if job_id:
            with job_lock:
                if job_id in processing_jobs:
                    processing_jobs[job_id]['processed_files'].append({
                        'filename': filename,
                        'status': 'failed',
                        'error': error_msg
                    })
                    processing_jobs[job_id]['failed'] += 1
        
        # Try to clean up local file even on error
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"✓ Cleaned up local file after error: {file_path}")
        except Exception as cleanup_error:
            print(f"⚠️  Failed to clean up {file_path}: {cleanup_error}")
        
        raise Exception(f"Failed to process {filename}: {error_msg}")


def process_document_async(
    file_path: Path,
    filename: str,
    metadata: Dict[str, Any],
    job_id: str,
    upload_to_s3: bool = False
):
    """Async wrapper for document processing"""
    try:
        process_document(file_path, filename, metadata, job_id, upload_to_s3)
    except Exception as e:
        print(f"Async processing failed: {e}")


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
        "message": "RAG API Server (Flask)",
        "version": "1.0.0",
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
        
        # Count documents and chunks
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
        import time
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 5)
        retrieval_k = data.get('retrieval_k', 20)
        use_reranking = data.get('use_reranking', True)
        document_type = data.get('document_type')
        department = data.get('department')
        min_score = data.get('min_score')
        
        rag = get_pipeline()
        start_time = time.time()
        
        # Query with filters if provided
        if document_type or department:
            # Custom query with metadata filtering
            conn = rag.vector_db.get_connection()
            cursor = conn.cursor()
            
            query_embedding = rag.embedder.embed_text(query_text)
            query_json = json.dumps(query_embedding.tolist())
            
            where_clauses = []
            params = [query_json]
            
            if document_type:
                where_clauses.append("JSON_EXTRACT(metadata, '$.document_type') = %s")
                params.append(document_type)
            
            if department:
                where_clauses.append("JSON_EXTRACT(metadata, '$.department') = %s")
                params.append(department)
            
            if min_score:
                where_clauses.append("DOT_PRODUCT(embedding, JSON_ARRAY_PACK(%s)) >= %s")
                params.append(min_score)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            params.append(top_k)
            
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
        document_type = request.args.get('document_type')
        department = request.args.get('department')
        
        # Reuse query endpoint logic
        request_data = {
            'query': query,
            'top_k': top_k,
            'document_type': document_type,
            'department': department
        }
        
        # Simulate POST request
        import time
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
    """
    Upload a single document
    
    Form data:
        - file: PDF file (required)
        - document_type: Document type (default: "document")
        - department: Department (default: "general")
        - tags: Comma-separated tags
        - author: Document author
        - project_id: Project ID
        - client_name: Client name
        - upload_to_s3: Whether to upload to S3 (default: based on config)
    """
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
        result = process_document(file_path, filename, metadata, upload_to_s3=upload_to_s3)
        
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
        # Cleanup on error
        if 'file_path' in locals() and Path(file_path).exists():
            try:
                Path(file_path).unlink()
            except:
                pass
        return jsonify({"error": str(e)}), 500


@app.route('/upload/batch', methods=['POST'])
def upload_batch():
    """
    Upload multiple documents at once
    
    Form data:
        - files: List of PDF files (required)
        - document_type: Document type for all files
        - department: Department for all files
        - tags: Comma-separated tags
        - author: Author
        - project_id: Project ID
        - client_name: Client name
        - async_processing: Process in background (default: false)
        - upload_to_s3: Upload to S3 (default: based on config)
    """
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
                "message": f"Batch processing started. Check /jobs/{job_id} for status",
                "files": [
                    {
                        "filename": filename,
                        "status": "queued",
                        "job_id": job_id
                    }
                    for _, filename in file_paths
                ]
            })
        else:
            # Process synchronously
            results = []
            for file_path, filename in file_paths:
                try:
                    result = process_document(
                        file_path,
                        filename,
                        metadata,
                        job_id,
                        upload_to_s3
                    )
                    
                    # Generate presigned URL if in S3
                    presigned_url = None
                    if result.get('s3_key') and upload_to_s3:
                        s3 = get_s3_manager()
                        if s3:
                            try:
                                presigned_url = s3.generate_presigned_url(result['s3_key'])
                            except Exception as e:
                                print(f"Failed to generate presigned URL: {e}")
                    
                    results.append({
                        "filename": filename,
                        "status": "success",
                        "chunks_created": result['chunks'],
                        "pages": result['pages'],
                        "s3_uploaded": bool(result.get('s3_uri')),
                        "download_url": presigned_url,
                        "local_file_deleted": result.get('local_file_deleted', False),
                        "message": "Processed successfully"
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "status": "failed",
                        "message": str(e)
                    })
            
            with job_lock:
                processing_jobs[job_id]['status'] = 'completed'
            
            return jsonify({
                "total_files": len(files),
                "successful": processing_jobs[job_id]['successful'],
                "failed": processing_jobs[job_id]['failed'],
                "job_id": job_id,
                "files": results
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
        if job_id not in processing_jobs:
            return jsonify({"error": "Job not found"}), 404
        
        return jsonify(processing_jobs[job_id])


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    with job_lock:
        return jsonify({
            "jobs": list(processing_jobs.values())
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
            JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.s3_uri')) as s3_uri,
            COUNT(*) as chunks
        FROM {CENTRAL_TABLE}
        WHERE {where_sql}
        GROUP BY source, doc_id, type, dept, pages, ingested, s3_uri
        ORDER BY ingested DESC
        LIMIT %s
        """
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        documents = []
        s3 = get_s3_manager()
        
        for row in rows:
            doc_info = {
                "source": row[0],
                "document_id": row[1],
                "type": row[2],
                "department": row[3],
                "pages": row[4],
                "ingestion_date": row[5],
                "s3_uri": row[6],
                "chunks": row[7]
            }
            
            # Add presigned URL if file is in S3
            if row[6] and s3:
                try:
                    # Extract S3 key from URI (s3://bucket/key)
                    s3_key = row[6].replace(f"s3://{S3_CONFIG['bucket']}/", "")
                    doc_info["download_url"] = s3.generate_presigned_url(s3_key)
                except Exception as e:
                    print(f"Failed to generate presigned URL: {e}")
            
            documents.append(doc_info)
        
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
        
        # Get S3 info before deleting
        cursor.execute(
            f"SELECT JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.s3_uri')) FROM {CENTRAL_TABLE} WHERE source = %s LIMIT 1",
            (source,)
        )
        row = cursor.fetchone()
        s3_uri = row[0] if row else None
        
        # Delete all chunks from this document
        cursor.execute(f"DELETE FROM {CENTRAL_TABLE} WHERE source = %s", (source,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if deleted_count == 0:
            return jsonify({"error": "Document not found"}), 404
        
        # Delete from S3 if it exists there
        s3_deleted = False
        if s3_uri:
            s3 = get_s3_manager()
            if s3:
                try:
                    s3_key = s3_uri.replace(f"s3://{S3_CONFIG['bucket']}/", "")
                    s3.delete_file(s3_key)
                    s3_deleted = True
                except Exception as e:
                    print(f"Failed to delete from S3: {e}")
        
        return jsonify({
            "message": f"Deleted {deleted_count} chunks from {source}",
            "deleted_chunks": deleted_count,
            "s3_deleted": s3_deleted
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Download Endpoint
# ============================================

@app.route('/documents/<source>/download', methods=['GET'])
def download_document(source: str):
    """
    Get download URL for a document
    If in S3, returns presigned URL
    """
    try:
        rag = get_pipeline()
        conn = rag.vector_db.get_connection()
        cursor = conn.cursor()
        
        # Get S3 URI
        cursor.execute(
            f"SELECT JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.s3_uri')) FROM {CENTRAL_TABLE} WHERE source = %s LIMIT 1",
            (source,)
        )
        row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not row or not row[0]:
            return jsonify({
                "error": "Document not found in S3. File may have been deleted locally."
            }), 404
        
        s3_uri = row[0]
        s3 = get_s3_manager()
        
        if not s3:
            return jsonify({
                "error": "S3 not configured"
            }), 503
        
        # Generate presigned URL
        s3_key = s3_uri.replace(f"s3://{S3_CONFIG['bucket']}/", "")
        presigned_url = s3.generate_presigned_url(s3_key, expiration=3600)  # 1 hour
        
        return jsonify({
            "source": source,
            "s3_uri": s3_uri,
            "download_url": presigned_url,
            "expires_in": 3600
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Initialize and Run
# ============================================

@app.before_first_request
def initialize():
    """Initialize on first request"""
    try:
        # Initialize pipeline
        rag = get_pipeline()
        print(f"✓ RAG Pipeline initialized")
        print(f"✓ Using centralized table: {CENTRAL_TABLE}")
        
        # Initialize S3 if enabled
        if S3_CONFIG['enabled']:
            s3 = get_s3_manager()
            if s3:
                print(f"✓ S3 integration enabled")
                print(f"  Bucket: {S3_CONFIG['bucket']}")
            else:
                print(f"⚠️  S3 enabled but initialization failed")
        else:
            print(f"ℹ️  S3 integration disabled")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG API Server (Flask)")
    print("=" * 60)
    print(f"Centralized Table: {CENTRAL_TABLE}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"S3 Enabled: {S3_CONFIG['enabled']}")
    if S3_CONFIG['enabled']:
        print(f"S3 Bucket: {S3_CONFIG['bucket']}")
    print("=" * 60)
    print()
    
    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        threaded=True
    )
