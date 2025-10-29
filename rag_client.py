"""
rag_client.py

Python client library for RAG API Server
Makes it easy to interact with the FastAPI RAG server
"""

import requests
from typing import List, Optional, Dict, Any
from pathlib import Path
import time


class RAGClient:
    """
    Python client for RAG API Server
    
    Usage:
        client = RAGClient("http://localhost:8000")
        
        # Upload a document
        result = client.upload("contract.pdf", document_type="contract")
        
        # Query documents
        results = client.query("What is the tax rate?")
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize RAG client
        
        Args:
            base_url: API server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and get statistics
        
        Returns:
            Dict with status, database info, document counts
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics
        
        Returns:
            Dict with totals, breakdowns by type/department, recent ingestions
        """
        response = requests.get(f"{self.base_url}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def upload(
        self,
        file_path: str,
        document_type: str = "document",
        department: str = "general",
        tags: str = "",
        author: Optional[str] = None,
        project_id: Optional[str] = None,
        client_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a single document
        
        Args:
            file_path: Path to PDF file
            document_type: Document type (contract, report, manual, etc.)
            department: Department (legal, engineering, sales, etc.)
            tags: Comma-separated tags
            author: Document author
            project_id: Related project ID
            client_name: Related client name
            
        Returns:
            Dict with filename, status, chunks_created, message
            
        Example:
            result = client.upload(
                "contract.pdf",
                document_type="contract",
                department="legal",
                tags="vendor,2024",
                client_name="Acme Corp"
            )
            print(f"Created {result['chunks_created']} chunks")
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError("Only PDF files are supported")
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            data = {
                "document_type": document_type,
                "department": department,
                "tags": tags
            }
            
            if author:
                data["author"] = author
            if project_id:
                data["project_id"] = project_id
            if client_name:
                data["client_name"] = client_name
            
            response = requests.post(
                f"{self.base_url}/upload",
                files=files,
                data=data,
                timeout=self.timeout
            )
        
        response.raise_for_status()
        return response.json()
    
    def upload_batch(
        self,
        file_paths: List[str],
        document_type: str = "document",
        department: str = "general",
        tags: str = "",
        author: Optional[str] = None,
        project_id: Optional[str] = None,
        client_name: Optional[str] = None,
        async_processing: bool = True,
        wait_for_completion: bool = False,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Upload multiple documents
        
        Args:
            file_paths: List of PDF file paths
            document_type: Document type for all files
            department: Department for all files
            tags: Comma-separated tags for all files
            author: Document author
            project_id: Related project ID
            client_name: Related client name
            async_processing: Process in background (recommended for >5 files)
            wait_for_completion: Wait for async job to complete (if async=True)
            poll_interval: Seconds between status checks (if waiting)
            
        Returns:
            Dict with total_files, successful, failed, job_id, files list
            
        Example:
            result = client.upload_batch(
                ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
                document_type="report",
                department="engineering",
                async_processing=True,
                wait_for_completion=True
            )
            print(f"Uploaded {result['successful']}/{result['total_files']} files")
        """
        # Validate files
        validated_paths = []
        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path.suffix.lower() == '.pdf':
                raise ValueError(f"Not a PDF file: {path}")
            validated_paths.append(path)
        
        # Prepare files
        files = [
            ("files", (path.name, open(path, "rb"), "application/pdf"))
            for path in validated_paths
        ]
        
        try:
            data = {
                "document_type": document_type,
                "department": department,
                "tags": tags,
                "async_processing": str(async_processing).lower()
            }
            
            if author:
                data["author"] = author
            if project_id:
                data["project_id"] = project_id
            if client_name:
                data["client_name"] = client_name
            
            response = requests.post(
                f"{self.base_url}/upload/batch",
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Wait for completion if requested
            if async_processing and wait_for_completion:
                job_id = result["job_id"]
                print(f"Waiting for job {job_id} to complete...")
                result = self.wait_for_job(job_id, poll_interval=poll_interval)
            
            return result
            
        finally:
            # Close all file handles
            for _, (_, f, _) in files:
                f.close()
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        use_reranking: bool = True,
        document_type: Optional[str] = None,
        department: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            retrieval_k: Number of candidates for reranking
            use_reranking: Apply reranking for better accuracy
            document_type: Filter by document type
            department: Filter by department
            min_score: Minimum relevance score (0-1)
            
        Returns:
            List of result dicts with content, source, page_num, score, metadata
            
        Example:
            results = client.query(
                "What is the corporate tax rate?",
                top_k=3,
                document_type="report"
            )
            
            for result in results:
                print(f"Score: {result['similarity_score']:.3f}")
                print(f"Source: {result['source']}, Page {result['page_num']}")
                print(f"Content: {result['content'][:200]}...")
        """
        request_data = {
            "query": query,
            "top_k": top_k,
            "retrieval_k": retrieval_k,
            "use_reranking": use_reranking
        }
        
        if document_type:
            request_data["document_type"] = document_type
        if department:
            request_data["department"] = department
        if min_score is not None:
            request_data["min_score"] = min_score
        
        response = requests.post(
            f"{self.base_url}/query",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_type: Optional[str] = None,
        department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple search (GET endpoint)
        
        Args:
            query: Search query
            top_k: Number of results
            document_type: Filter by document type
            department: Filter by department
            
        Returns:
            List of results
        """
        params = {"q": query, "top_k": top_k}
        
        if document_type:
            params["document_type"] = document_type
        if department:
            params["department"] = department
        
        response = requests.get(
            f"{self.base_url}/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def list_documents(
        self,
        document_type: Optional[str] = None,
        department: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List all documents
        
        Args:
            document_type: Filter by document type
            department: Filter by department
            limit: Maximum number of results
            
        Returns:
            List of document dicts with source, type, department, pages, chunks
            
        Example:
            docs = client.list_documents(document_type="contract")
            for doc in docs:
                print(f"{doc['source']}: {doc['chunks']} chunks, {doc['pages']} pages")
        """
        params = {"limit": limit}
        
        if document_type:
            params["document_type"] = document_type
        if department:
            params["department"] = department
        
        response = requests.get(
            f"{self.base_url}/documents",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["documents"]
    
    def delete_document(self, source: str) -> Dict[str, Any]:
        """
        Delete a document by source filename
        
        Args:
            source: Document filename (e.g., "contract.pdf")
            
        Returns:
            Dict with message and deleted_chunks count
            
        Example:
            result = client.delete_document("old_contract.pdf")
            print(f"Deleted {result['deleted_chunks']} chunks")
        """
        response = requests.delete(
            f"{self.base_url}/documents/{source}",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a batch processing job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict with job status, progress, processed files
        """
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Wait for an async job to complete
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            verbose: Print progress updates
            
        Returns:
            Final job status dict
            
        Example:
            result = client.upload_batch(files, async_processing=True)
            job_id = result["job_id"]
            final_status = client.wait_for_job(job_id)
            print(f"Completed: {final_status['successful']}/{final_status['total_files']}")
        """
        while True:
            status = self.get_job_status(job_id)
            
            if verbose:
                print(f"Job {job_id}: {status['status']} - "
                      f"{status['successful']}/{status['total_files']} completed "
                      f"({status['failed']} failed)")
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            time.sleep(poll_interval)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all processing jobs
        
        Returns:
            List of job status dicts
        """
        response = requests.get(f"{self.base_url}/jobs", timeout=10)
        response.raise_for_status()
        return response.json()["jobs"]


# ============================================
# Example Usage
# ============================================

def example_basic_usage():
    """Basic usage examples"""
    
    # Initialize client
    client = RAGClient("http://localhost:8000")
    
    # Check health
    health = client.health_check()
    print(f"✓ API is {health['status']}")
    print(f"  Documents: {health['documents']}, Chunks: {health['chunks']}")
    
    # Upload a document
    print("\nUploading document...")
    result = client.upload(
        "contract.pdf",
        document_type="contract",
        department="legal",
        tags="vendor,2024",
        client_name="Acme Corp"
    )
    print(f"✓ Uploaded: {result['filename']}")
    print(f"  Created {result['chunks_created']} chunks")
    
    # Query documents
    print("\nQuerying...")
    results = client.query(
        "What are the payment terms?",
        top_k=3,
        document_type="contract"
    )
    
    print(f"✓ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        score = result.get('rerank_score') or result.get('similarity_score', 0)
        print(f"\n  {i}. Score: {score:.3f}")
        print(f"     Source: {result['source']}, Page {result['page_num']}")
        print(f"     {result['content'][:150]}...")


def example_batch_upload():
    """Batch upload example"""
    
    client = RAGClient("http://localhost:8000")
    
    # Upload multiple files
    print("Uploading batch...")
    result = client.upload_batch(
        ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        document_type="report",
        department="engineering",
        async_processing=True,
        wait_for_completion=True
    )
    
    print(f"\n✓ Batch complete!")
    print(f"  Successful: {result['successful']}/{result['total_files']}")
    print(f"  Failed: {result['failed']}")


def example_document_management():
    """Document management example"""
    
    client = RAGClient("http://localhost:8000")
    
    # List all documents
    docs = client.list_documents()
    print(f"Total documents: {len(docs)}")
    
    # List contracts only
    contracts = client.list_documents(document_type="contract")
    print(f"Contracts: {len(contracts)}")
    
    for doc in contracts[:5]:
        print(f"  - {doc['source']}: {doc['chunks']} chunks")
    
    # Delete a document
    # result = client.delete_document("old_contract.pdf")
    # print(f"Deleted {result['deleted_chunks']} chunks")


if __name__ == "__main__":
    print("RAG Client Examples")
    print("=" * 60)
    
    # Run examples (uncomment as needed)
    # example_basic_usage()
    # example_batch_upload()
    # example_document_management()
    
    print("\nImport this module to use the RAGClient class:")
    print("  from rag_client import RAGClient")
    print("  client = RAGClient('http://localhost:8000')")
