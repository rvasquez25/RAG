"""
Simple RAG integration for agentic workflows
Use this for most projects
"""

import os
from advanced_rag import create_production_pipeline

class RAGTool:
    """RAG tool for your agent to call"""
    
    def __init__(self):
        # Configuration
        self.config = {
            'host': os.getenv('SINGLESTORE_HOST', 'localhost'),
            'port': int(os.getenv('SINGLESTORE_PORT', 3306)),
            'user': os.getenv('SINGLESTORE_USER', 'root'),
            'password': os.getenv('SINGLESTORE_PASSWORD', ''),
            'database': os.getenv('SINGLESTORE_DATABASE', 'rag_db')
        }
        
        # Initialize pipeline once (reuse for all queries)
        self.pipeline = create_production_pipeline(
            self.config,
            use_reranker=True,    # ✓ Use reranker for max accuracy
            use_hybrid=True       # ✓ Use hybrid search for better recall
        )
    
    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search documents and return formatted context
        
        Args:
            query: User's question or search query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Formatted context string for the agent/LLM
        """
        # Query with all accuracy features
        results = self.pipeline.query(
            query,
            top_k=top_k,
            retrieval_k=20,      # Retrieve more for reranking
            use_reranking=True   # Final refinement
        )
        
        # Format results for LLM
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Document: {result['source']}, Page: {result['page_num']}")
            context_parts.append(result['content'])
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    def search_with_metadata(
        self,
        query: str,
        document_type: str = None,
        department: str = None,
        top_k: int = 5
    ) -> str:
        """
        Search with metadata filtering
        Useful if you've organized documents with metadata
        """
        # You'll need to implement metadata filtering
        # See metadata_organization_examples.py for details
        pass


# Initialize once at startup
rag_tool = RAGTool()

# Use in your agent
def agent_step(user_query: str):
    """Example agent step"""
    
    # 1. Agent decides it needs to search documents
    if should_search_documents(user_query):
        context = rag_tool.search(user_query, top_k=5)
        
        # 2. Pass context to LLM
        prompt = f"""
        Based on the following information from our knowledge base:
        
        {context}
        
        User question: {user_query}
        
        Provide a helpful answer based on the information above.
        """
        
        answer = llm.generate(prompt)
        return answer
    
    # ... rest of agent logic
