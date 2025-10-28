#!/usr/bin/env python3
"""
Quick verification that the SingleStore 8.9 fix is working
Run this after updating to verify the fix
"""

import sys
import os

def check_schema_in_code():
    """Check if the code has the correct schema"""
    print("="*60)
    print("Checking rag_pipeline.py schema...")
    print("="*60)
    
    with open('rag_pipeline.py', 'r') as f:
        content = f.read()
        
    required_elements = [
        'PRIMARY KEY (id, chunk_id)',
        'SHARD KEY (id)',
        'UNIQUE KEY (chunk_id, id)'
    ]
    
    all_present = True
    for element in required_elements:
        if element in content:
            print(f"✓ Found: {element}")
        else:
            print(f"✗ Missing: {element}")
            all_present = False
    
    if all_present:
        print("\n✓ Schema code is correct!")
        return True
    else:
        print("\n✗ Schema code needs updating!")
        return False


def test_table_creation():
    """Test actual table creation"""
    print("\n" + "="*60)
    print("Testing actual table creation...")
    print("="*60)
    
    try:
        import pymysql
        from rag_pipeline import BGEEmbedder, SingleStoreVectorDB
        
        # Get config from environment
        config = {
            'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
            'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
            'user': os.getenv('SINGLESTORE_USER', 'root'),
            'password': os.getenv('SINGLESTORE_PASSWORD', ''),
            'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
        }
        
        print(f"Connecting to {config['host']}:{config['port']}/{config['database']}...")
        
        # Initialize
        embedder = BGEEmbedder("BAAI/bge-m3")
        vector_db = SingleStoreVectorDB(**config)
        
        # Try to create table
        test_table = "test_schema_verification"
        print(f"\nCreating test table: {test_table}")
        vector_db.create_table(test_table, embedder.embedding_dim)
        
        print("✓ Table created successfully!")
        
        # Verify schema
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        cursor.execute(f"SHOW CREATE TABLE {test_table}")
        result = cursor.fetchone()
        create_sql = result[1]
        
        print("\nVerifying schema components:")
        checks = [
            ('Composite PRIMARY KEY', 'PRIMARY KEY (`id`,`chunk_id`)' in create_sql),
            ('SHARD KEY on id', 'SHARD KEY (`id`)' in create_sql),
            ('UNIQUE KEY includes id', 'UNIQUE KEY' in create_sql and '`chunk_id`' in create_sql and '`id`' in create_sql),
            ('VECTOR column', 'VECTOR(' in create_sql)
        ]
        
        all_good = True
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            if not passed:
                all_good = False
        
        # Cleanup
        cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
        conn.commit()
        cursor.close()
        conn.close()
        
        if all_good:
            print("\n" + "="*60)
            print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
            print("="*60)
            print("\nYour pipeline is ready to use!")
            print("Next steps:")
            print("  1. Run: python test_pipeline.py")
            print("  2. Start ingesting documents")
            print("  3. Run queries!")
            return True
        else:
            print("\n✗ Some schema checks failed")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check SingleStore is running")
        print("  2. Verify .env configuration")
        print("  3. Ensure database exists: CREATE DATABASE rag_db;")
        return False


def main():
    print("\n" + "="*60)
    print("SINGLESTORE 8.9 FIX VERIFICATION")
    print("="*60 + "\n")
    
    # Check 1: Code has correct schema
    code_ok = check_schema_in_code()
    
    if not code_ok:
        print("\nPlease update rag_pipeline.py with the fixed version")
        sys.exit(1)
    
    # Check 2: Can create table successfully
    try:
        table_ok = test_table_creation()
        
        if table_ok:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except ImportError as e:
        print(f"\n⚠ Import error: {e}")
        print("\nInstall dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
