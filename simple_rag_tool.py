"""
simple_rag_tool.py

Production-ready RAG tool for agentic workflows
Copy this file and use it directly in your agent project

Usage:
    from simple_rag_tool import RAGTool
    
    rag = RAGTool()  # Initialize once
    context = rag.search("your query")  # Use many times
"""

import os
from typing import Optional, List, Dict, Any
from advanced_rag import create_production_pipeline


class RAGTool:
    """
    Simple RAG interface for agents
    
    Features:
    - Maximum accuracy (hybrid search + reranking)
    - Simple query interface
    - Formatted output for LLMs
    - Error handling built-in
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None
    ):
        """
        Initialize RAG tool
        
        Args:
            Database config (or uses environment variables)
            Set via: export SINGLESTORE_HOST=localhost, etc.
        """
        # Get config from args or environment
        config = {
            'host': host or os.getenv('SINGLESTORE_HOST', 'localhost'),
            'port': port or int(os.getenv('SINGLESTORE_PORT', 3306)),
            'user': user or os.getenv('SINGLESTORE_USER', 'root'),
            'password': password or os.getenv('SINGLESTORE_PASSWORD', ''),
            'database': database or os.getenv('SINGLESTORE_DATABASE', 'rag_db')
        }
        
        # Initialize with best settings for accuracy
        self.pipeline = create_production_pipeline(
            config,
            use_reranker=True,   # Max accuracy
            use_hybrid=True      # Better recall
        )
        
        print("✓ RAG Tool initialized")
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        min_relevance: float = None
    ) -> str:
        """
        Search for relevant information
        
        Args:
            query: User's question or search query
            num_results: How many sources to retrieve (default 5)
            min_relevance: Minimum relevance score 0-1 (default None = all)
            
        Returns:
            Formatted context string ready for your LLM
            
        Example:
            context = rag.search("What is the tax rate?")
            prompt = f"Based on: {context}\n\nAnswer: {user_query}"
        """
        try:
            # Query with all accuracy features
            results = self.pipeline.query(
                query,
                top_k=num_results,
                retrieval_k=20,  # Good for reranking
                use_reranking=True,
                score_threshold=min_relevance
            )
            
            # Format for LLM
            return self._format_results(results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def search_detailed(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search and get detailed results (for advanced use)
        
        Returns list of dicts with:
            - content: The text chunk
            - source: Document filename
            - page: Page number
            - score: Relevance score
        """
        try:
            results = self.pipeline.query(
                query,
                top_k=num_results,
                retrieval_k=20,
                use_reranking=True
            )
            
            return [
                {
                    'content': r['content'],
                    'source': r['source'],
                    'page': r.get('page_num', 'N/A'),
                    'score': r.get('rerank_score') or r.get('similarity_score', 0)
                }
                for r in results
            ]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results for LLM consumption"""
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Simple, clean format
        parts = []
        for i, result in enumerate(results, 1):
            parts.append(f"[Source {i}]")
            parts.append(result['content'])
            parts.append("")  # Blank line
        
        return "\n".join(parts)
    
    def test_connection(self) -> bool:
        """Test if RAG system is working"""
        try:
            self.pipeline.query("test", top_k=1, use_reranking=False)
            return True
        except Exception as e:
            print(f"RAG system not available: {e}")
            return False


# ============================================
# Example Usage
# ============================================

def example_basic_usage():
    """Simplest possible usage"""
    
    # 1. Initialize
    rag = RAGTool()
    
    # 2. Search
    context = rag.search("What is the corporate tax rate in Singapore?")
    
    # 3. Use in your agent/LLM
    print("Retrieved context:")
    print(context)
    
    # Build your LLM prompt
    prompt = f"""
    Use this information to answer the question:
    
    {context}
    
    Question: What is the corporate tax rate in Singapore?
    
    Answer:
    """
    
    # Send to your LLM
    # answer = your_llm(prompt)


def example_with_your_agent():
    """Example integration with agent workflow"""
    
    # Initialize RAG tool once
    rag = RAGTool()
    
    # Check it's working
    if not rag.test_connection():
        print("Warning: RAG not available, continuing without it")
        # Your agent can still work without RAG
    
    def agent_step(user_input: str) -> str:
        """Your agent's step function"""
        
        # Agent decides if it needs to search documents
        if needs_document_search(user_input):
            # Get relevant context
            context = rag.search(user_input, num_results=3)
            
            # Build enhanced prompt
            enhanced_prompt = f"""
            You have access to the following information:
            
            {context}
            
            User: {user_input}
            
            Assistant: Based on the information above,
            """
            
            # Generate response with context
            return your_llm_call(enhanced_prompt)
        else:
            # Regular agent response without RAG
            return your_llm_call(user_input)
    
    # Use agent
    response = agent_step("What is IRAS?")
    print(response)


def example_detailed_usage():
    """Example using detailed results (with sources)"""
    
    rag = RAGTool()
    
    # Get detailed results
    query = "What is the transfer pricing documentation requirement?"
    results = rag.search_detailed(query, num_results=3)
    
    # Build answer with citations
    answer_parts = []
    answer_parts.append("Based on the following sources:\n")
    
    for i, result in enumerate(results, 1):
        answer_parts.append(
            f"{i}. {result['source']} (Page {result['page']}) "
            f"- Relevance: {result['score']:.2f}"
        )
    
    answer_parts.append("\nInformation:")
    for result in results:
        answer_parts.append(f"- {result['content'][:200]}...")
    
    print("\n".join(answer_parts))


def example_error_handling():
    """Example with proper error handling"""
    
    try:
        rag = RAGTool()
        
        # Test connection first
        if not rag.test_connection():
            print("RAG system unavailable - using fallback")
            # Your fallback logic here
            return
        
        # Normal usage
        context = rag.search("your query")
        
        if "Error" in context or "No relevant" in context:
            # Handle no results case
            print("No information found, asking user for clarification")
        else:
            # Process context
            print(f"Found context: {len(context)} characters")
            
    except Exception as e:
        print(f"RAG initialization failed: {e}")
        # Your agent can continue without RAG


# ============================================
# Quick Test
# ============================================

def quick_test():
    """Quick test to verify everything works"""
    print("Testing RAG Tool...")
    
    try:
        # Initialize
        rag = RAGTool()
        print("✓ Initialized")
        
        # Test connection
        if rag.test_connection():
            print("✓ Connection OK")
        else:
            print("✗ Connection failed")
            return
        
        # Test query
        context = rag.search("test query", num_results=2)
        if context and not context.startswith("Error"):
            print("✓ Query works")
            print(f"  Retrieved {len(context)} characters")
        else:
            print("✗ Query failed")
            print(f"  {context}")
        
        print("\n✓ RAG Tool is ready to use!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nMake sure:")
        print("1. SingleStore is running")
        print("2. Database has documents ingested")
        print("3. Environment variables are set")


if __name__ == "__main__":
    # Run quick test
    quick_test()
    
    # Uncomment to run examples
    # example_basic_usage()
    # example_with_your_agent()
    # example_detailed_usage()
