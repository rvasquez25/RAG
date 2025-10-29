"""
Advanced RAG Pipeline with Hybrid Search and Reranking
For maximum retrieval accuracy
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import pymysql
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
import logging
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval
    for improved accuracy
    """
    
    def __init__(self, vector_db, embedder, alpha: float = 0.5):
        """
        Initialize hybrid retriever
        
        Args:
            vector_db: SingleStoreVectorDB instance
            embedder: BGEEmbedder instance
            alpha: Weight for dense retrieval (1-alpha for sparse)
                  0.5 = equal weight, 1.0 = only dense, 0.0 = only sparse
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.alpha = alpha
    
    def _compute_bm25_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[float]:
        """
        Compute BM25 scores for documents
        
        Args:
            query: Query string
            documents: List of document dictionaries
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            List of BM25 scores
        """
        # Tokenize query
        query_terms = self._tokenize(query.lower())
        
        # Tokenize documents and compute stats
        doc_tokens = [self._tokenize(doc['content'].lower()) for doc in documents]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        # Compute document frequencies
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        
        N = len(documents)
        scores = []
        
        for doc_idx, tokens in enumerate(doc_tokens):
            score = 0.0
            doc_length = doc_lengths[doc_idx]
            term_freqs = Counter(tokens)
            
            for term in query_terms:
                if term in term_freqs:
                    # IDF component
                    idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                    
                    # TF component
                    tf = term_freqs[term]
                    norm = k1 * (1 - b + b * (doc_length / avg_doc_length))
                    tf_component = (tf * (k1 + 1)) / (tf + norm)
                    
                    score += idf * tf_component
            
            scores.append(score)
        
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter short tokens
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def search(
        self,
        table_name: str,
        query_text: str,
        top_k: int = 20,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            table_name: Database table name
            query_text: Query string
            top_k: Number of candidates to retrieve (should be > final_top_k)
            final_top_k: Final number of results to return
            
        Returns:
            List of top results with hybrid scores
        """
        # Get dense retrieval results (more than needed for hybrid)
        query_embedding = self.embedder.embed_text(query_text)
        dense_results = self.vector_db.search(
            table_name,
            query_embedding,
            top_k=top_k
        )
        
        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query_text, dense_results)
        
        # Normalize both score types
        dense_scores = [r['similarity_score'] for r in dense_results]
        normalized_dense = self._normalize_scores(dense_scores)
        normalized_bm25 = self._normalize_scores(bm25_scores)
        
        # Combine scores
        for i, result in enumerate(dense_results):
            hybrid_score = (
                self.alpha * normalized_dense[i] +
                (1 - self.alpha) * normalized_bm25[i]
            )
            result['hybrid_score'] = hybrid_score
            result['dense_score'] = dense_scores[i]
            result['bm25_score'] = bm25_scores[i]
        
        # Sort by hybrid score and return top results
        dense_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return dense_results[:final_top_k]


class Reranker:
    """
    Cross-encoder reranker for final result refinement
    Provides highest accuracy but is slower
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace model identifier for reranker
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'content' field
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of documents with rerank scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k:
            return documents[:top_k]
        return documents


class AdvancedRAGPipeline:
    """
    Advanced RAG Pipeline with hybrid search and reranking
    Optimized for maximum accuracy
    """
    
    def __init__(
        self,
        embedder,
        vector_db,
        reranker: Optional[Reranker] = None,
        table_name: str = "document_embeddings",
        chunk_size: int = 512,
        overlap: int = 128,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.5
    ):
        """
        Initialize advanced RAG pipeline
        
        Args:
            embedder: BGEEmbedder instance
            vector_db: SingleStoreVectorDB instance
            reranker: Optional Reranker instance
            table_name: Database table name
            chunk_size: Text chunk size
            overlap: Chunk overlap size
            use_hybrid: Whether to use hybrid retrieval
            hybrid_alpha: Weight for dense vs sparse (if use_hybrid=True)
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.reranker = reranker
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            self.retriever = HybridRetriever(
                vector_db,
                embedder,
                alpha=hybrid_alpha
            )
        
        # Create table
        self.vector_db.create_table(table_name, embedder.embedding_dim)
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        use_reranking: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query with advanced retrieval strategies
        
        Args:
            query_text: Query string
            top_k: Final number of results
            retrieval_k: Initial retrieval size (for reranking)
            use_reranking: Whether to apply reranking
            score_threshold: Minimum score threshold
            
        Returns:
            List of top results with metadata
        """
        logger.info(f"Processing query: {query_text}")
        
        # Step 1: Initial retrieval
        if self.use_hybrid:
            results = self.retriever.search(
                self.table_name,
                query_text,
                top_k=retrieval_k,
                final_top_k=retrieval_k if use_reranking else top_k
            )
        else:
            query_embedding = self.embedder.embed_text(query_text)
            results = self.vector_db.search(
                self.table_name,
                query_embedding,
                top_k=retrieval_k if use_reranking else top_k
            )
        
        # Step 2: Reranking (optional but recommended)
        if use_reranking and self.reranker and results:
            results = self.reranker.rerank(query_text, results, top_k=top_k)
        elif not use_reranking and len(results) > top_k:
            results = results[:top_k]
        
        # Step 3: Apply score threshold
        if score_threshold is not None:
            score_key = 'rerank_score' if use_reranking else 'hybrid_score' if self.use_hybrid else 'similarity_score'
            results = [r for r in results if r.get(score_key, 0) >= score_threshold]
        
        logger.info(f"Returning {len(results)} results")
        return results
    
    def query_with_context(
        self,
        query_text: str,
        top_k: int = 5,
        context_window: int = 1
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Query and return results with surrounding context chunks
        
        Args:
            query_text: Query string
            top_k: Number of results
            context_window: Number of chunks before/after to include
            
        Returns:
            Tuple of (results, formatted_context_string)
        """
        results = self.query(query_text, top_k=top_k)
        
        # Get surrounding chunks for each result
        enhanced_results = []
        for result in results:
            # Query for chunks from same source/page
            context_chunks = self._get_context_chunks(
                result['source'],
                result['page_num'],
                result['chunk_index'],
                context_window
            )
            
            result['context_chunks'] = context_chunks
            enhanced_results.append(result)
        
        # Format context for LLM
        formatted_context = self._format_context_for_llm(enhanced_results)
        
        return enhanced_results, formatted_context
    
    def _get_context_chunks(
        self,
        source: str,
        page_num: int,
        chunk_index: int,
        window: int
    ) -> List[Dict[str, Any]]:
        """Get surrounding chunks for context"""
        conn = pymysql.connect(**self.vector_db.connection_params)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        try:
            query = f"""
            SELECT content, chunk_index
            FROM {self.table_name}
            WHERE source = %s 
            AND page_num = %s
            AND chunk_index BETWEEN %s AND %s
            ORDER BY chunk_index
            """
            
            cursor.execute(
                query,
                (source, page_num, chunk_index - window, chunk_index + window)
            )
            return cursor.fetchall()
            
        finally:
            cursor.close()
            conn.close()
    
    def _format_context_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """Format results into context string for LLM"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Source: {result['source']}, Page: {result['page_num']}")
            
            if 'context_chunks' in result:
                full_content = " ".join([
                    chunk['content'] 
                    for chunk in sorted(result['context_chunks'], key=lambda x: x['chunk_index'])
                ])
                context_parts.append(full_content)
            else:
                context_parts.append(result['content'])
            
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)


class QueryExpansion:
    """
    Query expansion techniques for improved recall
    """
    
    @staticmethod
    def expand_with_synonyms(query: str, embedder) -> List[str]:
        """
        Generate query variations using embedding similarity
        
        Args:
            query: Original query
            embedder: BGEEmbedder instance
            
        Returns:
            List of query variations
        """
        # This is a placeholder - in production, you might use:
        # - WordNet for synonyms
        # - A language model for paraphrasing
        # - Domain-specific thesaurus
        
        variations = [query]
        
        # Simple word replacement strategy
        words = query.lower().split()
        
        # Add variations with different word orders for short queries
        if len(words) <= 5:
            variations.append(" ".join(reversed(words)))
        
        return variations


def create_production_pipeline(
    singlestore_config: Dict[str, Any],
    use_reranker: bool = True,
    use_hybrid: bool = True,
    table_name: str = "document_embeddings"
) -> AdvancedRAGPipeline:
    """
    Factory function to create production-ready pipeline
    
    Args:
        singlestore_config: Database configuration dictionary
        use_reranker: Whether to enable reranking
        use_hybrid: Whether to use hybrid search
        table_name: Centralized table name (default: document_embeddings)
        
    Returns:
        Configured AdvancedRAGPipeline instance
    """
    from rag_pipeline import BGEEmbedder, SingleStoreVectorDB
    
    # Initialize embedder
    embedder = BGEEmbedder("BAAI/bge-m3")
    
    # Initialize vector database
    vector_db = SingleStoreVectorDB(**singlestore_config)
    
    # Ensure centralized table exists
    vector_db.create_table(table_name, embedder.embedding_dim)
    
    # Initialize reranker if requested
    reranker = Reranker() if use_reranker else None
    
    # Create pipeline
    pipeline = AdvancedRAGPipeline(
        embedder=embedder,
        vector_db=vector_db,
        reranker=reranker,
        use_hybrid=use_hybrid,
        hybrid_alpha=0.5,  # Equal weight to dense and sparse
        chunk_size=512,
        overlap=128,
        table_name=table_name
    )
    
    return pipeline


def main():
    """Example usage"""
    
    SINGLESTORE_CONFIG = {
        'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
        'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
        'user': os.getenv('SINGLESTORE_USER', 'root'),
        'password': os.getenv('SINGLESTORE_PASSWORD', ''),
        'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
    }
    
    # Create advanced pipeline
    pipeline = create_production_pipeline(
        SINGLESTORE_CONFIG,
        use_reranker=True,
        use_hybrid=True
    )
    
    # Query with all accuracy enhancements
    results, formatted_context = pipeline.query_with_context(
        "What are the key findings?",
        top_k=5,
        context_window=1
    )
    
    # Display results
    print("\n=== Search Results ===\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result.get('rerank_score', result.get('hybrid_score', result.get('similarity_score'))):.4f}")
        print(f"  Source: {result['source']}, Page: {result['page_num']}")
        print(f"  Content: {result['content'][:200]}...")
        print()
    
    print("\n=== Formatted Context for LLM ===\n")
    print(formatted_context[:500] + "...")


if __name__ == "__main__":
    main()
