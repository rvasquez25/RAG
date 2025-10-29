"""
Document Processor Module
Handles document ingestion pipeline: extraction, chunking, embedding, and storage
"""

import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from rag_pipeline import PDFExtractor, ChunkingStrategy


class DocumentProcessor:
    """
    Processes documents through the complete RAG ingestion pipeline
    
    Pipeline stages:
    1. PDF text extraction
    2. Semantic chunking
    3. Metadata enrichment
    4. Embedding generation
    5. Database storage
    6. Optional S3 upload
    7. Local file cleanup
    """
    
    def __init__(
        self,
        rag_pipeline,
        s3_manager=None,
        table_name: str = "document_embeddings",
        auto_cleanup: bool = True
    ):
        """
        Initialize document processor
        
        Args:
            rag_pipeline: AdvancedRAGPipeline instance
            s3_manager: Optional S3Manager instance
            table_name: Database table name
            auto_cleanup: Automatically delete local files after processing
        """
        self.rag_pipeline = rag_pipeline
        self.s3_manager = s3_manager
        self.table_name = table_name
        self.auto_cleanup = auto_cleanup
        
        print(f"âœ“ DocumentProcessor initialized")
        print(f"  Table: {table_name}")
        print(f"  S3 enabled: {s3_manager is not None}")
        print(f"  Auto cleanup: {auto_cleanup}")
    
    def process_document(
        self,
        file_path: Path,
        filename: str,
        metadata: Dict[str, Any],
        upload_to_s3: bool = False,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline
        
        Args:
            file_path: Local file path
            filename: Original filename
            metadata: Document metadata
            upload_to_s3: Whether to upload to S3
            job_id: Optional job ID for tracking
            
        Returns:
            Result dictionary with processing information
            
        Raises:
            Exception: If processing fails
        """
        s3_uri = None
        s3_key = None
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print(f"{'='*60}")
            
            # Step 1: Extract text from PDF
            print(f"[1/6] Extracting text from PDF...")
            documents = PDFExtractor.extract_text(str(file_path))
            
            if not documents:
                raise ValueError("No text could be extracted from PDF")
            
            print(f"  âœ“ Extracted {len(documents)} pages")
            
            # Step 2: Chunk documents
            print(f"[2/6] Chunking document...")
            chunked_docs = ChunkingStrategy.chunk_documents(documents)
            print(f"  âœ“ Created {len(chunked_docs)} chunks")
            
            # Step 3: Upload to S3 FIRST (so we can include S3 info in metadata)
            if upload_to_s3 and self.s3_manager:
                print(f"[3/6] Uploading to S3...")
                s3_key, s3_uri = self._upload_to_s3(
                    file_path,
                    filename,
                    metadata
                )
                print(f"  ✓ Uploaded to S3: {s3_uri}")
            else:
                print(f"[3/6] Skipping S3 upload")
                s3_key = None
                s3_uri = None
            
            # Step 4: Enrich metadata (including S3 info)
            print(f"[4/6] Enriching metadata...")
            file_metadata = self._create_file_metadata(
                file_path,
                filename,
                len(documents),
                len(chunked_docs),
                metadata,
                s3_key=s3_key,
                s3_uri=s3_uri
            )
            
            # Add metadata to all chunks
            for chunk in chunked_docs:
                chunk.metadata = {**chunk.metadata, **file_metadata}
            
            print(f"  ✓ Metadata added to {len(chunked_docs)} chunks")
            if s3_uri:
                print(f"    Including S3 location: {s3_uri}")
            
            # Step 5: Generate embeddings
            print(f"[5/6] Generating embeddings...")
            texts = [doc.content for doc in chunked_docs]
            embeddings = self.rag_pipeline.embedder.embed_batch(texts)
            print(f"  ✓ Generated {len(embeddings)} embeddings")
            
            # Step 6: Store in database (with S3 metadata already included)
            print(f"[6/6] Storing in database...")
            self.rag_pipeline.vector_db.insert_embeddings(
                self.table_name,
                chunked_docs,
                embeddings
            )
            print(f"  ✓ Stored {len(chunked_docs)} chunks in {self.table_name}")
            
            # Cleanup local file
            if self.auto_cleanup:
                self._cleanup_local_file(file_path)
            
            # Build result
            result = {
                'filename': filename,
                'status': 'success',
                'chunks': len(chunked_docs),
                'pages': len(documents),
                's3_uri': s3_uri,
                's3_key': s3_key,
                'local_file_deleted': self.auto_cleanup,
                'processing_time': None  # Can be added if timing is tracked
            }
            
            print(f"\nâœ“ Successfully processed {filename}")
            print(f"  Pages: {result['pages']}, Chunks: {result['chunks']}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nâœ— Failed to process {filename}: {error_msg}")
            traceback.print_exc()
            
            # Cleanup on error
            if self.auto_cleanup:
                self._cleanup_local_file(file_path)
            
            raise Exception(f"Failed to process {filename}: {error_msg}")
    
    def _create_file_metadata(
        self,
        file_path: Path,
        filename: str,
        total_pages: int,
        total_chunks: int,
        user_metadata: Dict[str, Any],
        s3_key: Optional[str] = None,
        s3_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the document
        
        Args:
            file_path: Path to the file
            filename: Original filename
            total_pages: Number of pages extracted
            total_chunks: Number of chunks created
            user_metadata: User-provided metadata
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "document_id": file_path.stem,
            "source_file": filename,
            "file_path": str(file_path),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "file_size_kb": file_path.stat().st_size // 1024 if file_path.exists() else 0,
            "ingestion_date": datetime.now().isoformat(),
            "s3_key": s3_key,
            "s3_uri": s3_uri,
            **user_metadata
        }
        
        return metadata
    
    def _upload_to_s3(
        self,
        file_path: Path,
        filename: str,
        metadata: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Upload file to S3 with metadata
        
        Args:
            file_path: Local file path
            filename: Original filename
            metadata: Document metadata
            
        Returns:
            Tuple of (s3_key, s3_uri)
        """
        if not self.s3_manager:
            raise ValueError("S3Manager not configured")
        
        # Generate S3 key with date-based organization
        from s3_manager import S3Manager
        s3_key = S3Manager.generate_s3_key(
            filename,
            prefix='documents',
            use_date_prefix=True
        )
        
        # Prepare S3 metadata
        s3_metadata = {
            'original-filename': filename,
            'document-id': file_path.stem,
            'upload-date': datetime.now().isoformat(),
            'document-type': metadata.get('document_type', 'document'),
            'department': metadata.get('department', 'general')
        }
        
        # Upload to S3
        s3_uri = self.s3_manager.upload_file(
            file_path,
            s3_key,
            metadata=s3_metadata
        )
        
        return s3_key, s3_uri
    
    def _cleanup_local_file(self, file_path: Path) -> bool:
        """
        Delete local file after processing
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"  âœ“ Deleted local file: {file_path.name}")
                return True
            return False
        except Exception as e:
            print(f"  âš ï¸  Failed to delete local file {file_path}: {e}")
            return False
    
    
    def _update_chunks_with_s3_metadata(
        self,
        source_filename: str,
        s3_key: str,
        s3_uri: str
    ) -> int:
        """
        Update all chunks for a document with S3 information
        
        Args:
            source_filename: Original filename (source field in database)
            s3_key: S3 object key
            s3_uri: Full S3 URI
            
        Returns:
            Number of chunks updated
        """
        import pymysql
        import json
        
        conn = pymysql.connect(**self.rag_pipeline.vector_db.connection_params)
        cursor = conn.cursor()
        
        try:
            # Update all chunks from this source with S3 info
            update_sql = f"""
            UPDATE {self.table_name}
            SET metadata = JSON_SET(
                metadata,
                '$.s3_key', %s,
                '$.s3_uri', %s
            )
            WHERE source = %s
            """
            
            cursor.execute(update_sql, (s3_key, s3_uri, source_filename))
            updated_count = cursor.rowcount
            
            conn.commit()
            return updated_count
            
        except Exception as e:
            print(f"  ⚠️  Failed to update chunks with S3 metadata: {e}")
            conn.rollback()
            return 0
        finally:
            cursor.close()
            conn.close()

    def process_batch(
        self,
        file_paths: list[tuple[Path, str]],
        metadata: Dict[str, Any],
        upload_to_s3: bool = False,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of (file_path, filename) tuples
            metadata: Shared metadata for all documents
            upload_to_s3: Whether to upload to S3
            job_id: Optional job ID for tracking
            
        Returns:
            Summary dict with successful/failed counts and results
        """
        results = {
            'total': len(file_paths),
            'successful': 0,
            'failed': 0,
            'files': []
        }
        
        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(file_paths)} files")
        print(f"{'='*60}\n")
        
        for i, (file_path, filename) in enumerate(file_paths, 1):
            print(f"\nFile {i}/{len(file_paths)}: {filename}")
            
            try:
                result = self.process_document(
                    file_path,
                    filename,
                    metadata,
                    upload_to_s3=upload_to_s3,
                    job_id=job_id
                )
                
                results['files'].append(result)
                results['successful'] += 1
                
            except Exception as e:
                error_result = {
                    'filename': filename,
                    'status': 'failed',
                    'error': str(e)
                }
                results['files'].append(error_result)
                results['failed'] += 1
                
                print(f"  âœ— Failed: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"{'='*60}\n")
        
        return results
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate file before processing
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        # Check if it's a PDF
        if not file_path.suffix.lower() == '.pdf':
            return False, f"Not a PDF file: {file_path}"
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_path.stat().st_size > max_size:
            return False, f"File too large (max 100MB): {file_path}"
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try reading first 1KB
        except Exception as e:
            return False, f"File not readable: {e}"
        
        return True, None


class ProcessingJobTracker:
    """
    Tracks document processing jobs
    
    Used for async batch processing to monitor progress
    """
    
    def __init__(self):
        """Initialize job tracker"""
        self.jobs = {}
    
    def create_job(self, job_id: str, total_files: int) -> Dict[str, Any]:
        """
        Create a new job
        
        Args:
            job_id: Unique job identifier
            total_files: Total number of files to process
            
        Returns:
            Job dict
        """
        job = {
            'job_id': job_id,
            'total_files': total_files,
            'successful': 0,
            'failed': 0,
            'status': 'pending',
            'current_file': None,
            'processed_files': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': None
        }
        
        self.jobs[job_id] = job
        return job
    
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_file: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ):
        """
        Update job status
        
        Args:
            job_id: Job identifier
            status: New status
            current_file: Currently processing file
            result: Processing result to add
        """
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        
        if status:
            job['status'] = status
        
        if current_file:
            job['current_file'] = current_file
        
        if result:
            job['processed_files'].append(result)
            
            if result.get('status') == 'success':
                job['successful'] += 1
            else:
                job['failed'] += 1
        
        # Mark as completed if all files processed
        if job['successful'] + job['failed'] >= job['total_files']:
            job['status'] = 'completed'
            job['completed_at'] = datetime.now().isoformat()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> list[Dict[str, Any]]:
        """Get all jobs"""
        return list(self.jobs.values())
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Remove old completed jobs
        
        Args:
            max_age_hours: Maximum age in hours to keep
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for job_id, job in self.jobs.items():
            if job['status'] == 'completed' and job.get('completed_at'):
                completed_time = datetime.fromisoformat(job['completed_at'])
                if completed_time < cutoff:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old jobs")


# Example usage
if __name__ == "__main__":
    from advanced_rag import create_production_pipeline
    import os
    
    # Configuration
    config = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Create pipeline
    pipeline = create_production_pipeline(config)
    
    # Create processor
    processor = DocumentProcessor(
        rag_pipeline=pipeline,
        s3_manager=None,  # No S3 for this example
        table_name="document_embeddings",
        auto_cleanup=False  # Don't delete for testing
    )
    
    print("\nâœ“ DocumentProcessor ready")
    print("  Use processor.process_document() to ingest PDFs")
