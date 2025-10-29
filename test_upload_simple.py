"""
test_upload_simple.py

Simple test script for FastAPI upload endpoints
Uses your existing PDF files
"""

import requests
import json
from pathlib import Path
from typing import List


# ============================================
# Configuration
# ============================================

BASE_URL = "http://localhost:8000"


# ============================================
# Test Functions
# ============================================

def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Table: {data['table']}")
            print(f"  Documents: {data['documents']}")
            print(f"  Chunks: {data['chunks']}")
            return True
        else:
            print(f"✗ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python fastapi_upload_server.py")
        return False


def test_upload_single(pdf_path: str):
    """
    Test single file upload
    
    Args:
        pdf_path: Path to PDF file
    """
    print("\n" + "=" * 60)
    print(f"Testing Single Upload: {pdf_path}")
    print("=" * 60)
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"✗ File not found: {pdf_path}")
        return False
    
    if not pdf_file.suffix.lower() == '.pdf':
        print(f"✗ Not a PDF file: {pdf_path}")
        return False
    
    try:
        print(f"\nUploading {pdf_file.name}...")
        
        with open(pdf_file, "rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            data = {
                "document_type": "test_document",
                "department": "testing",
                "tags": "test,api_test",
                "author": "Test Script"
            }
            
            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                data=data,
                timeout=60
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Upload successful")
            print(f"  Filename: {result['filename']}")
            print(f"  Status: {result['status']}")
            print(f"  Chunks created: {result['chunks_created']}")
            print(f"  Pages: {result['pages']}")
            print(f"  Message: {result['message']}")
            return True
        else:
            print(f"\n✗ Upload failed")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_upload_batch(pdf_paths: List[str]):
    """
    Test batch file upload
    
    Args:
        pdf_paths: List of paths to PDF files
    """
    print("\n" + "=" * 60)
    print(f"Testing Batch Upload: {len(pdf_paths)} files")
    print("=" * 60)
    
    # Validate files
    valid_files = []
    for path in pdf_paths:
        pdf_file = Path(path)
        if not pdf_file.exists():
            print(f"⚠ Skipping (not found): {path}")
            continue
        if not pdf_file.suffix.lower() == '.pdf':
            print(f"⚠ Skipping (not PDF): {path}")
            continue
        valid_files.append(pdf_file)
    
    if not valid_files:
        print("\n✗ No valid PDF files found")
        return False
    
    print(f"\nUploading {len(valid_files)} files...")
    for f in valid_files:
        print(f"  - {f.name}")
    
    try:
        # Open all files
        files = [
            ("files", (f.name, open(f, "rb"), "application/pdf"))
            for f in valid_files
        ]
        
        data = {
            "document_type": "batch_test",
            "department": "testing",
            "tags": "test,batch,api_test",
            "project_id": "TEST-BATCH-001"
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/upload/batch",
                files=files,
                data=data,
                timeout=120
            )
            
            print(f"\nStatus: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✓ Batch upload completed")
                print(f"  Total files: {result['total_files']}")
                print(f"  Successful: {result['successful']}")
                print(f"  Failed: {result['failed']}")
                
                print("\nFile details:")
                for file_result in result['files']:
                    status_icon = "✓" if file_result['status'] == 'success' else "✗"
                    print(f"  {status_icon} {file_result['filename']}")
                    if file_result['status'] == 'success':
                        print(f"     Chunks: {file_result['chunks_created']}, Pages: {file_result['pages']}")
                    else:
                        print(f"     Error: {file_result.get('message', 'Unknown error')}")
                
                return result['failed'] == 0
            else:
                print(f"\n✗ Batch upload failed")
                print(f"  Response: {response.text}")
                return False
                
        finally:
            # Close all file handles
            for _, (_, f, _) in files:
                f.close()
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "=" * 60)
    print("Testing Statistics")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Statistics retrieved")
            print(f"  Total documents: {data['total_documents']}")
            print(f"  Total chunks: {data['total_chunks']}")
            
            if data['by_type']:
                print("\n  Documents by type:")
                for item in data['by_type']:
                    print(f"    {item['type']}: {item['count']}")
            
            if data['by_department']:
                print("\n  Documents by department:")
                for item in data['by_department']:
                    print(f"    {item['department']}: {item['count']}")
            
            return True
        else:
            print(f"\n✗ Failed to get statistics")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# ============================================
# Main
# ============================================

def main():
    """Main entry point"""
    import sys
    
    print("=" * 60)
    print("  RAG Upload API - Simple Test Script")
    print("=" * 60)
    print(f"\nServer: {BASE_URL}")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_upload_simple.py health")
        print("  python test_upload_simple.py upload <file.pdf>")
        print("  python test_upload_simple.py batch <file1.pdf> <file2.pdf> ...")
        print("  python test_upload_simple.py stats")
        print("\nExamples:")
        print("  python test_upload_simple.py health")
        print("  python test_upload_simple.py upload document.pdf")
        print("  python test_upload_simple.py batch doc1.pdf doc2.pdf doc3.pdf")
        print("  python test_upload_simple.py stats")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Run command
    if command == "health":
        success = test_health()
    
    elif command == "upload":
        if len(sys.argv) < 3:
            print("\n✗ Error: Please specify a PDF file")
            print("  Usage: python test_upload_simple.py upload <file.pdf>")
            sys.exit(1)
        success = test_upload_single(sys.argv[2])
    
    elif command == "batch":
        if len(sys.argv) < 3:
            print("\n✗ Error: Please specify PDF files")
            print("  Usage: python test_upload_simple.py batch <file1.pdf> <file2.pdf> ...")
            sys.exit(1)
        pdf_files = sys.argv[2:]
        success = test_upload_batch(pdf_files)
    
    elif command == "stats":
        success = test_stats()
    
    else:
        print(f"\n✗ Unknown command: {command}")
        print("\nValid commands: health, upload, batch, stats")
        sys.exit(1)
    
    # Exit with appropriate code
    print()
    if success:
        print("✓ Test completed successfully")
        sys.exit(0)
    else:
        print("✗ Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
