"""
Flask RAG Server Test Client
Simple script to test Flask server endpoints
"""

import requests
import json
from pathlib import Path
import time


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Server is healthy")
            print(f"  Database: {data['database']}")
            print(f"  Table: {data['table']}")
            print(f"  Documents: {data['documents']}")
            print(f"  Chunks: {data['chunks']}")
            print(f"  S3 Enabled: {data['s3_enabled']}")
            return True
        else:
            print(f"✗ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python flask_server.py")
        return False


def test_upload(pdf_path: str, upload_to_s3: bool = False):
    """Test single file upload"""
    print("\n" + "="*60)
    print(f"Testing Upload: {pdf_path}")
    print(f"S3 Upload: {upload_to_s3}")
    print("="*60)
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"✗ File not found: {pdf_path}")
        return False
    
    try:
        print(f"\nUploading {pdf_file.name}...")
        
        with open(pdf_file, "rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            data = {
                "document_type": "test_document",
                "department": "testing",
                "tags": "test,flask,api",
                "author": "Test Script",
                "upload_to_s3": str(upload_to_s3).lower()
            }
            
            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                data=data,
                timeout=120
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Upload successful")
            print(f"  Filename: {result['filename']}")
            print(f"  Status: {result['status']}")
            print(f"  Chunks: {result['chunks_created']}")
            print(f"  Pages: {result['pages']}")
            print(f"  S3 Uploaded: {result.get('s3_uploaded', False)}")
            print(f"  Local Deleted: {result.get('local_file_deleted', False)}")
            
            if result.get('s3_uploaded'):
                print(f"  S3 URI: {result.get('s3_uri')}")
                if result.get('download_url'):
                    print(f"  Download URL: {result['download_url'][:80]}...")
            
            return True
        else:
            print(f"\n✗ Upload failed")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_query(query: str, top_k: int = 3):
    """Test query endpoint"""
    print("\n" + "="*60)
    print(f"Testing Query: {query}")
    print("="*60)
    
    try:
        data = {
            "query": query,
            "top_k": top_k,
            "use_reranking": True
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Query successful")
            print(f"  Results: {result['total_results']}")
            print(f"  Time: {result['retrieval_time_ms']:.1f}ms")
            
            for i, res in enumerate(result['results'][:3], 1):
                score = res.get('rerank_score') or res.get('similarity_score', 0)
                print(f"\n  Result {i}:")
                print(f"    Score: {score:.4f}")
                print(f"    Source: {res['source']}, Page: {res['page_num']}")
                print(f"    Content: {res['content'][:100]}...")
            
            return True
        else:
            print(f"\n✗ Query failed")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_list_documents():
    """Test document listing"""
    print("\n" + "="*60)
    print("Testing Document Listing")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/documents", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Listed {result['total']} documents")
            
            for doc in result['documents'][:5]:
                print(f"\n  {doc['source']}:")
                print(f"    Type: {doc['type']}, Dept: {doc['department']}")
                print(f"    Pages: {doc['pages']}, Chunks: {doc['chunks']}")
                print(f"    Ingested: {doc['ingestion_date']}")
                
                if doc.get('s3_uri'):
                    print(f"    S3: {doc['s3_uri']}")
                if doc.get('download_url'):
                    print(f"    Download: Available")
            
            return True
        else:
            print(f"\n✗ Listing failed")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_download_url(source: str):
    """Test getting download URL for a document"""
    print("\n" + "="*60)
    print(f"Testing Download URL: {source}")
    print("="*60)
    
    try:
        response = requests.get(
            f"{BASE_URL}/documents/{source}/download",
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Download URL generated")
            print(f"  S3 URI: {result['s3_uri']}")
            print(f"  Expires in: {result['expires_in']}s")
            print(f"  URL: {result['download_url'][:80]}...")
            
            # Test if URL is accessible
            print("\n  Testing URL accessibility...")
            test_response = requests.head(result['download_url'], timeout=5)
            if test_response.status_code in [200, 302]:
                print("  ✓ URL is accessible")
            else:
                print(f"  ✗ URL returned {test_response.status_code}")
            
            return True
        else:
            print(f"\n✗ Failed to get download URL")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*60)
    print("Testing Statistics")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Statistics retrieved")
            print(f"  Total Documents: {data['total_documents']}")
            print(f"  Total Chunks: {data['total_chunks']}")
            
            if data['by_type']:
                print("\n  By Type:")
                for item in data['by_type'][:5]:
                    print(f"    {item['type']}: {item['count']}")
            
            if data['by_department']:
                print("\n  By Department:")
                for item in data['by_department'][:5]:
                    print(f"    {item['department']}: {item['count']}")
            
            return True
        else:
            print(f"\n✗ Statistics failed")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def run_full_test_suite(pdf_path: str = None, use_s3: bool = False):
    """Run complete test suite"""
    print("\n" + "="*60)
    print("  Flask RAG Server - Full Test Suite")
    print("="*60)
    print(f"\nServer: {BASE_URL}")
    print(f"S3 Testing: {use_s3}")
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health()
    if not results['health']:
        print("\n✗ Server not healthy. Stopping tests.")
        return results
    
    time.sleep(1)
    
    # Test 2: Upload (if PDF provided)
    if pdf_path:
        results['upload'] = test_upload(pdf_path, upload_to_s3=use_s3)
        time.sleep(2)
    else:
        print("\n⚠️  No PDF provided, skipping upload test")
        results['upload'] = None
    
    # Test 3: Statistics
    results['stats'] = test_stats()
    time.sleep(1)
    
    # Test 4: List documents
    results['list'] = test_list_documents()
    time.sleep(1)
    
    # Test 5: Query
    results['query'] = test_query("test query", top_k=3)
    time.sleep(1)
    
    # Test 6: Download URL (only if S3 enabled and document uploaded)
    if use_s3 and pdf_path and results.get('upload'):
        results['download'] = test_download_url(Path(pdf_path).name)
    else:
        results['download'] = None
    
    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name.capitalize():15s} {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    return results


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python flask_client_test.py health")
        print("  python flask_client_test.py upload <file.pdf> [--s3]")
        print("  python flask_client_test.py query '<query text>'")
        print("  python flask_client_test.py list")
        print("  python flask_client_test.py stats")
        print("  python flask_client_test.py download <filename>")
        print("  python flask_client_test.py full <file.pdf> [--s3]")
        print("\nExamples:")
        print("  python flask_client_test.py health")
        print("  python flask_client_test.py upload document.pdf --s3")
        print("  python flask_client_test.py query 'what is the tax rate'")
        print("  python flask_client_test.py full document.pdf --s3")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "health":
        success = test_health()
    
    elif command == "upload":
        if len(sys.argv) < 3:
            print("✗ Please specify PDF file")
            sys.exit(1)
        use_s3 = '--s3' in sys.argv
        success = test_upload(sys.argv[2], upload_to_s3=use_s3)
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("✗ Please specify query text")
            sys.exit(1)
        success = test_query(sys.argv[2])
    
    elif command == "list":
        success = test_list_documents()
    
    elif command == "stats":
        success = test_stats()
    
    elif command == "download":
        if len(sys.argv) < 3:
            print("✗ Please specify filename")
            sys.exit(1)
        success = test_download_url(sys.argv[2])
    
    elif command == "full":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else None
        use_s3 = '--s3' in sys.argv
        results = run_full_test_suite(pdf_path, use_s3=use_s3)
        success = all(r for r in results.values() if r is not None)
    
    else:
        print(f"✗ Unknown command: {command}")
        sys.exit(1)
    
    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
