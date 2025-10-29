"""
S3 Manager Module
Handles all S3-related operations including uploads, downloads, and presigned URLs
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
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = 'us-east-1',
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize S3 client
        
        Args:
            bucket: S3 bucket name
            region: AWS region (default: us-east-1)
            access_key: AWS access key (uses env var if None)
            secret_key: AWS secret key (uses env var if None)
        """
        if not S3_AVAILABLE:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        self.bucket = bucket
        self.region = region
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
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
                
                # Create bucket with appropriate configuration
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                print(f"✓ Created S3 bucket: {self.bucket}")
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
    
    Returns:
        Configuration dict with bucket, region, credentials, enabled flag
    """
    return {
        'bucket': os.getenv('S3_BUCKET', 'my-rag-documents'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
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
            secret_key=config.get('secret_key')
        )
        return manager
    except Exception as e:
        print(f"⚠️  Failed to initialize S3: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Test S3 manager
    config = get_s3_config_from_env()
    
    if config['enabled'] and S3_AVAILABLE:
        manager = create_s3_manager_from_config(config)
        
        if manager:
            print("\n✓ S3Manager test successful")
            print(f"  Bucket: {manager.bucket}")
            print(f"  Region: {manager.region}")
            
            # List files
            files = manager.list_files(prefix='documents', max_keys=10)
            print(f"\n  Files in bucket: {len(files)}")
            for f in files[:5]:
                print(f"    - {f}")
    else:
        print("S3 not enabled or boto3 not available")
