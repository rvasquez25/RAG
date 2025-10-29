"""
S3 Manager Module
Handles all S3-related operations including uploads, downloads, and presigned URLs
Supports AWS S3 and S3-compatible storage (MinIO, DigitalOcean Spaces, etc.)
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Optional: Import boto3 for S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class S3Manager:
    """
    Manages S3 operations for document storage
    
    Features:
    - Upload files to S3 with metadata
    - Generate presigned URLs for secure downloads
    - Delete files from S3
    - Automatic bucket creation if needed
    - Supports AWS S3 and S3-compatible storage
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = 'us-east-1',
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize S3 client
        
        Args:
            bucket: S3 bucket name
            region: AWS region (default: us-east-1, optional for S3-compatible storage)
            access_key: AWS access key (uses AWS_ACCESS_KEY_ID env var if None)
            secret_key: AWS secret key (uses AWS_SECRET_ACCESS_KEY env var if None)
            endpoint_url: Custom S3 endpoint for S3-compatible storage (MinIO, DigitalOcean Spaces, etc.)
        
        Example for AWS S3:
            manager = S3Manager(bucket='my-bucket', region='us-east-1')
        
        Example for MinIO:
            manager = S3Manager(
                bucket='my-bucket',
                access_key='minioadmin',
                secret_key='minioadmin',
                endpoint_url='http://localhost:9000'
            )
        
        Example for DigitalOcean Spaces:
            manager = S3Manager(
                bucket='my-space',
                region='nyc3',
                endpoint_url='https://nyc3.digitaloceanspaces.com'
            )
        """
        if not S3_AVAILABLE:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        
        # Build client configuration
        client_config = {}
        
        # Add credentials
        access_key_value = access_key or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key_value = secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if access_key_value:
            client_config['aws_access_key_id'] = access_key_value
        if secret_key_value:
            client_config['aws_secret_access_key'] = secret_key_value
        
        # Add region (required for AWS S3, optional for S3-compatible)
        if region:
            client_config['region_name'] = region
        
        # Add endpoint_url for S3-compatible storage (MinIO, DigitalOcean Spaces, etc.)
        if endpoint_url:
            client_config['endpoint_url'] = endpoint_url
            print(f"Using S3-compatible storage endpoint: {endpoint_url}")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', **client_config)
        except Exception as e:
            print(f"Failed to initialize S3 client: {e}")
            raise
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        print(f"✓ S3Manager initialized for bucket: {bucket}")
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            print(f"✓ Connected to existing S3 bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Creating S3 bucket: {self.bucket}")
                
                try:
                    # For S3-compatible storage or us-east-1, use simple create
                    if self.endpoint_url or self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket)
                    else:
                        # For AWS S3 in other regions, specify location constraint
                        self.s3_client.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    print(f"✓ Created S3 bucket: {self.bucket}")
                except ClientError as create_error:
                    # If bucket creation fails, it might already exist or we lack permissions
                    # Try to use it anyway
                    print(f"⚠️  Bucket creation returned error (may already exist): {create_error}")
                    print(f"   Attempting to use bucket anyway...")
            else:
                raise
    
    def upload_file(
        self,
        file_path: Path,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
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
            
            # Add metadata if provided
            if metadata:
                # S3 metadata keys must be lowercase and use hyphens
                extra_args['Metadata'] = {
                    k.lower().replace(' ', '-').replace('_', '-'): str(v) 
                    for k, v in metadata.items()
                }
            
            # Upload file
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
            Presigned URL string
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
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            print(f"✓ Deleted from S3: {s3_key}")
            return True
        except Exception as e:
            print(f"✗ S3 deletion failed: {e}")
            raise
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if file exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def get_file_metadata(self, s3_key: str) -> Optional[Dict[str, str]]:
        """
        Get metadata for a file in S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Metadata dict or None if file doesn't exist
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return response.get('Metadata', {})
        except ClientError:
            return None
    
    def list_files(self, prefix: str = '', max_keys: int = 1000) -> list:
        """
        List files in S3 bucket with optional prefix
        
        Args:
            prefix: S3 key prefix to filter by
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"✗ Failed to list S3 files: {e}")
            return []
    
    @staticmethod
    def generate_s3_key(
        filename: str,
        prefix: str = 'documents',
        use_date_prefix: bool = True
    ) -> str:
        """
        Generate standardized S3 key for a file
        
        Args:
            filename: Original filename
            prefix: Base prefix (default: 'documents')
            use_date_prefix: Include date in path (default: True)
            
        Returns:
            S3 key string
        """
        if use_date_prefix:
            date_prefix = datetime.now().strftime('%Y/%m/%d')
            return f"{prefix}/{date_prefix}/{filename}"
        else:
            return f"{prefix}/{filename}"


def get_s3_config_from_env() -> Dict[str, any]:
    """
    Load S3 configuration from environment variables
    
    Environment variables:
    - S3_BUCKET: Bucket name (default: my-rag-documents)
    - AWS_REGION: AWS region (default: us-east-1)
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key
    - S3_ENDPOINT_URL: Custom endpoint for S3-compatible storage (MinIO, DigitalOcean Spaces, etc.)
    - ENABLE_S3: Enable S3 integration (default: false)
    
    Returns:
        Configuration dict with bucket, region, credentials, endpoint_url, enabled flag
    """
    return {
        'bucket': os.getenv('S3_BUCKET', 'my-rag-documents'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'endpoint_url': os.getenv('S3_ENDPOINT_URL'),  # For MinIO, DigitalOcean Spaces, etc.
        'enabled': os.getenv('ENABLE_S3', 'false').lower() == 'true'
    }


def create_s3_manager_from_config(config: Dict[str, any]) -> Optional[S3Manager]:
    """
    Create S3Manager instance from configuration
    
    Args:
        config: Configuration dict from get_s3_config_from_env()
        
    Returns:
        S3Manager instance or None if S3 not available/enabled
    """
    if not config.get('enabled'):
        print("ℹ️  S3 integration disabled")
        return None
    
    if not S3_AVAILABLE:
        print("⚠️  S3 enabled but boto3 not installed")
        print("   Install with: pip install boto3")
        return None
    
    try:
        manager = S3Manager(
            bucket=config['bucket'],
            region=config['region'],
            access_key=config.get('access_key'),
            secret_key=config.get('secret_key'),
            endpoint_url=config.get('endpoint_url')
        )
        return manager
    except Exception as e:
        print(f"⚠️  Failed to initialize S3: {e}")
        return None


# Example usage and tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("S3 Manager - Configuration Test")
    print("="*60 + "\n")
    
    # Test 1: Load config from environment
    config = get_s3_config_from_env()
    
    print("Configuration loaded from environment:")
    print(f"  S3 Enabled: {config['enabled']}")
    print(f"  Bucket: {config['bucket']}")
    print(f"  Region: {config['region']}")
    print(f"  Endpoint URL: {config.get('endpoint_url', 'None (using AWS S3)')}")
    print(f"  Access Key: {'***' if config.get('access_key') else 'Not set'}")
    print(f"  Secret Key: {'***' if config.get('secret_key') else 'Not set'}")
    
    if not S3_AVAILABLE:
        print("\n⚠️  boto3 not installed")
        print("   Install with: pip install boto3")
        exit(1)
    
    if config['enabled']:
        print("\n" + "="*60)
        print("Testing S3 Connection")
        print("="*60 + "\n")
        
        try:
            manager = create_s3_manager_from_config(config)
            
            if manager:
                print("\n✓ S3Manager initialized successfully")
                print(f"  Bucket: {manager.bucket}")
                print(f"  Region: {manager.region}")
                
                # Test listing files
                print("\nListing files in bucket...")
                files = manager.list_files(prefix='documents', max_keys=10)
                print(f"  Found {len(files)} files")
                for i, f in enumerate(files[:5], 1):
                    print(f"    {i}. {f}")
                
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
                
                print("\n✓ All tests passed!")
            else:
                print("\n✗ Failed to initialize S3Manager")
        
        except Exception as e:
            print(f"\n✗ S3 test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nℹ️  S3 integration disabled")
        print("   Set ENABLE_S3=true to enable")
        print("\nExample configuration:")
        print("  export ENABLE_S3=true")
        print("  export S3_BUCKET=my-bucket")
        print("  export AWS_ACCESS_KEY_ID=your-access-key")
        print("  export AWS_SECRET_ACCESS_KEY=your-secret-key")
        print("  export S3_ENDPOINT_URL=https://your-endpoint  # Optional, for S3-compatible storage")
