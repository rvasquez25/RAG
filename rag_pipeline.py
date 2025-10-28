"""
High-Accuracy RAG Pipeline with BGE-M3 and SingleStore
Handles PDF extraction, chunking, embedding, and retrieval
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import pymysql
from sentence_transformers import SentenceTransformer
import PyPDF2
from pathlib import Path
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    source: str
    page_num: Optional[int] = None
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PDFExtractor:
    """Extract text from PDF documents with high fidelity"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> List[Document]:
        """
        Extract text from PDF with page-level granularity
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects, one per page
        """
        documents = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    
                    # Skip empty pages
                    if text.strip():
                        doc = Document(
                            content=text.strip(),
                            source=pdf_path,
                            page_num=page_num,
                            metadata={
                                'total_pages': len(pdf_reader.pages),
                                'file_name': Path(pdf_path).name
                            }
                        )
                        documents.append(doc)
                        
            logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            raise


class ChunkingStrategy:
    """Advanced chunking strategies for optimal RAG performance"""
    
    @staticmethod
    def semantic_chunking(
        text: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[str]:
        """
        Chunk text with semantic awareness and overlap
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        # Split on paragraph boundaries first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If single paragraph exceeds chunk_size, split it
            if para_length > chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = para.split('. ')
                temp_chunk = []
                temp_length = 0
                
                for sent in sentences:
                    sent = sent.strip() + '.'
                    sent_length = len(sent)
                    
                    if temp_length + sent_length > chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        # Keep last sentence for overlap
                        temp_chunk = [temp_chunk[-1]] if overlap > 0 else []
                        temp_length = len(temp_chunk[0]) if temp_chunk else 0
                    
                    temp_chunk.append(sent)
                    temp_length += sent_length
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    
            # Normal paragraph handling
            elif current_length + para_length > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last paragraph for overlap if it's small enough
                    if overlap > 0 and len(current_chunk[-1]) < overlap:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[-1])
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(para)
                current_length += para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def chunk_documents(
        documents: List[Document],
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[Document]:
        """
        Chunk a list of documents
        
        Args:
            documents: List of Document objects
            chunk_size: Target chunk size
            overlap: Overlap size
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = ChunkingStrategy.semantic_chunking(
                doc.content,
                chunk_size,
                overlap
            )
            
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc.source}_{doc.page_num}_{i}".encode()
                ).hexdigest()
                
                chunked_doc = Document(
                    content=chunk,
                    source=doc.source,
                    page_num=doc.page_num,
                    chunk_id=chunk_id,
                    metadata={
                        **(doc.metadata or {}),
                        'chunk_index': i,
                        'total_chunks_in_page': len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs


class BGEEmbedder:
    """BGE-M3 embedding model wrapper"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize BGE-M3 embedder
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts efficiently
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            Array of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings


class SingleStoreVectorDB:
    """SingleStore vector database interface"""
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str
    ):
        """
        Initialize SingleStore connection
        
        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
        self.database = database
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = pymysql.connect(**self.connection_params)
            conn.close()
            logger.info("Successfully connected to SingleStore")
        except Exception as e:
            logger.error(f"Failed to connect to SingleStore: {e}")
            raise
    
    def create_table(self, table_name: str, embedding_dim: int):
        """
        Create a table for storing embeddings (SingleStore 8.9+ compatible)
        
        Args:
            table_name: Name of the table
            embedding_dim: Dimension of embedding vectors
        """
        conn = pymysql.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            # Drop existing table if needed
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table optimized for SingleStore 8.9+
            # Using id as shard key and chunk_id as unique key that includes id
            create_table_sql = f"""
            CREATE TABLE {table_name} (
                id BIGINT AUTO_INCREMENT,
                chunk_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                source VARCHAR(512) NOT NULL,
                page_num INT,
                chunk_index INT,
                metadata JSON,
                embedding VECTOR({embedding_dim}) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, chunk_id),
                SHARD KEY (id),
                UNIQUE KEY (chunk_id, id),
                KEY (source),
                KEY (page_num),
                KEY (source, page_num)
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info(f"Created table {table_name} with {embedding_dim}-dim vectors")
            logger.info(f"Table uses composite primary key (id, chunk_id) with shard key on id")
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_embeddings(
        self,
        table_name: str,
        documents: List[Document],
        embeddings: np.ndarray
    ):
        """
        Insert documents and their embeddings
        
        Args:
            table_name: Target table name
            documents: List of Document objects
            embeddings: Corresponding embedding vectors
        """
        conn = pymysql.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            insert_sql = f"""
            INSERT INTO {table_name} 
            (chunk_id, content, source, page_num, chunk_index, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            content = VALUES(content),
            source = VALUES(source),
            page_num = VALUES(page_num),
            chunk_index = VALUES(chunk_index),
            metadata = VALUES(metadata),
            embedding = VALUES(embedding)
            """
            
            import json
            
            for doc, emb in zip(documents, embeddings):
                # Convert embedding to list for storage
                emb_list = emb.tolist()
                emb_json = json.dumps(emb_list)
                
                metadata_json = json.dumps(doc.metadata) if doc.metadata else None
                
                cursor.execute(insert_sql, (
                    doc.chunk_id,
                    doc.content,
                    doc.source,
                    doc.page_num,
                    doc.metadata.get('chunk_index') if doc.metadata else None,
                    metadata_json,
                    emb_json
                ))
            
            conn.commit()
            logger.info(f"Inserted {len(documents)} documents into {table_name}")
            
        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def search(
        self,
        table_name: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            table_name: Table to search
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of result dictionaries with content and metadata
        """
        conn = pymysql.connect(**self.connection_params)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        try:
            import json
            query_json = json.dumps(query_embedding.tolist())
            
            # Use DOT_PRODUCT for normalized vectors (equivalent to cosine similarity)
            search_sql = f"""
            SELECT 
                content,
                source,
                page_num,
                chunk_index,
                metadata,
                DOT_PRODUCT(embedding, JSON_ARRAY_PACK(%s)) as similarity_score
            FROM {table_name}
            ORDER BY similarity_score DESC
            LIMIT %s
            """
            
            cursor.execute(search_sql, (query_json, top_k))
            results = cursor.fetchall()
            
            # Filter by threshold if provided
            if score_threshold is not None:
                results = [r for r in results if r['similarity_score'] >= score_threshold]
            
            # Parse metadata JSON
            for result in results:
                if result['metadata']:
                    result['metadata'] = json.loads(result['metadata'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
        finally:
            cursor.close()
            conn.close()


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_db: SingleStoreVectorDB,
        table_name: str = "document_embeddings",
        chunk_size: int = 512,
        overlap: int = 128
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedder: BGEEmbedder instance
            vector_db: SingleStoreVectorDB instance
            table_name: Name for the embeddings table
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Create table if needed
        self.vector_db.create_table(table_name, embedder.embedding_dim)
    
    def ingest_pdf(self, pdf_path: str):
        """
        Ingest a PDF document into the pipeline
        
        Args:
            pdf_path: Path to PDF file
        """
        logger.info(f"Ingesting PDF: {pdf_path}")
        
        # Extract text from PDF
        documents = PDFExtractor.extract_text(pdf_path)
        
        # Chunk documents
        chunked_docs = ChunkingStrategy.chunk_documents(
            documents,
            self.chunk_size,
            self.overlap
        )
        
        # Generate embeddings
        texts = [doc.content for doc in chunked_docs]
        embeddings = self.embedder.embed_batch(texts)
        
        # Store in database
        self.vector_db.insert_embeddings(
            self.table_name,
            chunked_docs,
            embeddings
        )
        
        logger.info(f"Successfully ingested {pdf_path}")
    
    def ingest_directory(self, directory_path: str):
        """
        Ingest all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
        """
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                self.ingest_pdf(str(pdf_file))
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_file}: {e}")
                continue
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant document chunks with metadata
        """
        logger.info(f"Querying: {query_text}")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query_text)
        
        # Search vector database
        results = self.vector_db.search(
            self.table_name,
            query_embedding,
            top_k,
            score_threshold
        )
        
        logger.info(f"Found {len(results)} results")
        return results


def main():
    """Example usage of the RAG pipeline"""
    
    # Configuration
    SINGLESTORE_CONFIG = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Initialize components
    embedder = BGEEmbedder("BAAI/bge-m3")
    vector_db = SingleStoreVectorDB(**SINGLESTORE_CONFIG)
    
    # Create pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_db=vector_db,
        table_name="document_embeddings",
        chunk_size=512,
        overlap=128
    )
    
    # Ingest documents
    # pipeline.ingest_pdf("path/to/document.pdf")
    # pipeline.ingest_directory("path/to/pdf_directory")
    
    # Query
    results = pipeline.query(
        "What is the main topic discussed?",
        top_k=5,
        score_threshold=0.5
    )
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['similarity_score']:.4f}) ---")
        print(f"Source: {result['source']}, Page: {result['page_num']}")
        print(f"Content: {result['content'][:200]}...")


if __name__ == "__main__":
    main()
