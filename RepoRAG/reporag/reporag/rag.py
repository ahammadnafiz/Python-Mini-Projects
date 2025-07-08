import logging
from typing import List, Dict, Any, Optional, Tuple
import shutil
import os
import hashlib
import pickle
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

class RepoRAG:
    """RAG system optimized for repository content."""

    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retrieval: int = 4,
        cache_dir: str = "./cache"
    ):
        self.setup_logging()
        self.initialize_components(
            google_api_key, model_name, embedding_model,
            chunk_size, chunk_overlap, k_retrieval, cache_dir
        )

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(
        self,
        google_api_key: str,
        model_name: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        k_retrieval: int,
        cache_dir: str
    ):
        """Initialize RAG components."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.cache_dir = cache_dir
        self.chat_history = []

        # Initialize text splitter optimized for code
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\nclass ",
                "\n\ndef ",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=model_name,
            temperature=0.1
        )

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize prompt templates
        self._initialize_prompts()

    def _initialize_prompts(self):
        """Initialize different prompt templates for different types of queries."""
        self.prompts = {
            "code": """You are a code analysis assistant. Focus on:
                1. Code structure and patterns
                2. Implementation details
                3. Best practices and potential improvements
                4. Dependencies and requirements

                Current context:
                {context}

                Question: {question}
                """,

            "documentation": """You are a documentation specialist. Focus on:
                1. Project overview and purpose
                2. Installation and setup
                3. Usage instructions
                4. API documentation

                Current context:
                {context}

                Question: {question}
                """,

            "general": """You are a repository analysis assistant. Focus on:
                1. Repository structure and organization
                2. Code and documentation quality
                3. Implementation patterns
                4. Project requirements

                Current context:
                {context}

                Question: {question}
                """
        }

    def ingest_content(self, content: str) -> None:
        """Ingest repository content with code-optimized processing."""
        try:
            # Clear existing vector store
            self.clear_vector_store()

            # Split content using RecursiveCharacterTextSplitter
            documents = self.text_splitter.create_documents([content])

            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "content_type": self._detect_content_type(doc.page_content),
                    "language": self._detect_language(doc.page_content),
                    "size": len(doc.page_content)
                })

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Initialize retrieval chain
            self._initialize_qa_chain()

            self.logger.info("Content ingestion completed successfully")

        except Exception as e:
            self.logger.error(f"Error during content ingestion: {str(e)}")
            raise

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content."""
        if any(keyword in content for keyword in ['def ', 'class ', 'import ', 'from ']):
            return "code"
        elif content.strip().startswith('#') or '```' in content:
            return "markdown"
        return "text"

    def _detect_language(self, content: str) -> str:
        """Detect programming language from content."""
        if any(keyword in content for keyword in ['def ', 'import ', '__init__']):
            return 'python'
        elif any(keyword in content for keyword in ['function ', 'const ', 'let ', 'var ']):
            return 'javascript'
        elif any(keyword in content for keyword in ['public class', 'private ', 'public static']):
            return 'java'
        elif any(keyword in content for keyword in ['#include', 'int main', 'printf']):
            return 'c'
        return 'unknown'

    def _initialize_qa_chain(self):
        """Initialize the QA chain with custom prompts and retrieval settings."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k_retrieval,
                    "filter": None
                }
            ),
            return_source_documents=True,
            verbose=False
        )

    def query(
        self,
        question: str,
        context_filter: Optional[Dict] = None,
        query_type: str = "general"
    ) -> Dict[str, Any]:
        """Query the RAG system with custom filtering and response generation."""
        if not self.vector_store:
            raise ValueError("No content ingested. Call ingest_content first.")

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(question, context_filter, query_type)

            # Check cache
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response

            # Update retriever filter
            self._update_retriever_filter(query_type, context_filter)

            # Get appropriate prompt
            prompt = self.prompts.get(query_type, self.prompts["general"])

            # Execute query
            response = self.qa_chain({
                "question": question,
                "chat_history": self.get_chat_history()
            })

            # Process and structure response
            # processed_response = self._process_response(response, query_type)

            # Update chat history
            # self._update_chat_history(question, processed_response["answer"])
            self._update_chat_history(question, response["answer"])
            
            # Cache response
            # self._cache_response(cache_key, processed_response)
            self._cache_response(cache_key, response)

            # return processed_response
            return response

        except Exception as e:
            self.logger.error(f"Error in query: {str(e)}")
            raise

    def _process_response(self, response: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Process and structure the response based on query type."""
        # Format the answer for terminal readability
        formatted_answer = self._format_for_terminal(response["answer"])

        processed_response = {
            "answer": formatted_answer,
            "sources": [],
            "metadata": {
                "query_type": query_type,
                "timestamp": RepoRAG._get_timestamp()
            }
        }

        # Extract and format source information
        for doc in response.get("source_documents", []):
            source_info = {
                "content": doc.page_content[:200] + "...",  # Truncate for brevity
                "metadata": doc.metadata
            }
            processed_response["sources"].append(source_info)

        # Create a formatted string representation
        processed_response["formatted_output"] = self._create_formatted_output(processed_response)

        return processed_response

    def _format_for_terminal(self, text: str) -> str:
        """Format text for better terminal readability."""
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Format code blocks
        text = re.sub(
            r'```(\w+)?\n(.*?)\n```',
            lambda m: f"\n{'=' * 40}\n{m.group(2)}\n{'=' * 40}\n",
            text,
            flags=re.DOTALL
        )

        # Format inline code
        text = re.sub(
            r'`([^`]+)`',
            lambda m: f"[{m.group(1)}]",
            text
        )

        return text

    def _create_formatted_output(self, processed_response: Dict[str, Any]) -> str:
        """Create a formatted string representation of the response."""
        sections = []

        # Add answer section
        sections.append("### Answer\n")
        sections.append(processed_response["answer"])

        # Add sources section if available
        if processed_response["sources"]:
            sections.append("\n### Sources\n")
            for idx, source in enumerate(processed_response["sources"], 1):
                source_text = f"\n{idx}. **File:** {source['metadata'].get('file_path', 'unknown')}"
                if 'language' in source['metadata']:
                    source_text += f"\n   **Language:** {source['metadata']['language']}"
                source_text += f"\n   **Preview:** {source['content']}"
                sections.append(source_text)

        # Add metadata section
        sections.append("\n### Metadata\n")
        sections.append(f"**Query Type:** {processed_response['metadata']['query_type']}")
        sections.append(f"**Timestamp:** {processed_response['metadata']['timestamp']}")

        return "\n".join(sections)

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _generate_cache_key(
        self,
        question: str,
        context_filter: Optional[Dict],
        query_type: str
    ) -> str:
        """Generate cache key including context filter and query type."""
        key_content = f"{question}-{str(context_filter)}-{query_type}"
        return hashlib.md5(key_content.encode('utf-8')).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if response is cached."""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response."""
        cache_path = os.path.join(self.cache_dir, cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(response, f)

    def _update_retriever_filter(self, query_type: str, custom_filter: Optional[Dict] = None):
        """Update retriever's filter based on query type and custom filter."""
        base_filter = {}

        # Add query type-specific filters
        if query_type == "code":
            base_filter["content_type"] = "code"
        elif query_type == "documentation":
            base_filter["content_type"] = "markdown"

        # Merge with custom filter if provided
        if custom_filter:
            base_filter.update(custom_filter)

        # Update retriever
        if base_filter:
            self.qa_chain.retriever.search_kwargs["filter"] = base_filter
        else:
            self.qa_chain.retriever.search_kwargs["filter"] = None

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Get formatted chat history."""
        return [(self.chat_history[i].content, self.chat_history[i + 1].content)
                for i in range(0, len(self.chat_history), 2)]

    def _update_chat_history(self, question: str, answer: str) -> None:
        """Update chat history."""
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

    def clear_memory(self) -> None:
        """Clear chat history."""
        self.chat_history = []
        self.logger.info("Chat history cleared")

    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        if hasattr(self, 'vector_store'):
            self.logger.info("Clearing vector store")
            self.vector_store = None

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        self.logger.info("Clearing cache directory")
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def clear_vector_database(self) -> None:
        """Clear the vector database completely to start fresh with a new repository."""
        try:
            self.logger.info("Clearing vector database for new repository")
            
            # Clear the in-memory vector store
            self.vector_store = None
            
            # Clear the QA chain
            if hasattr(self, 'qa_chain'):
                self.qa_chain = None
            
            # Clear chat history (optional - you might want to keep this)
            self.clear_memory()
            
            # Clear cache (optional - you might want to keep this for performance)
            self.clear_cache()
            
            self.logger.info("Vector database cleared successfully - ready for new repository")
            
        except Exception as e:
            self.logger.error(f"Error clearing vector database: {e}")
            raise

    def switch_repository(self, content: str, clear_history: bool = True, clear_cache: bool = False) -> None:
        """Switch to a new repository by clearing previous data and ingesting new content.
        
        Args:
            content: The new repository content to ingest
            clear_history: Whether to clear chat history (default: True)
            clear_cache: Whether to clear response cache (default: False for performance)
        """
        try:
            self.logger.info("Switching to new repository")
            
            # Clear vector store
            self.clear_vector_store()
            
            # Clear QA chain
            if hasattr(self, 'qa_chain'):
                self.qa_chain = None
            
            # Optionally clear chat history
            if clear_history:
                self.clear_memory()
                self.logger.info("Chat history cleared")
            
            # Optionally clear cache
            if clear_cache:
                self.clear_cache()
                self.logger.info("Response cache cleared")
            
            # Ingest new content
            self.ingest_content(content)
            
            self.logger.info("Successfully switched to new repository")
            
        except Exception as e:
            self.logger.error(f"Error switching repository: {e}")
            raise

    def get_repository_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded repository."""
        info = {
            "has_vector_store": self.vector_store is not None,
            "has_qa_chain": hasattr(self, 'qa_chain') and self.qa_chain is not None,
            "chat_history_length": len(self.chat_history) // 2,
            "cache_size": 0
        }
        
        # Get cache information
        if os.path.exists(self.cache_dir):
            cache_size = 0
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    cache_size += os.path.getsize(os.path.join(root, file))
            info["cache_size"] = cache_size
        
        # Get vector store stats if available
        if self.vector_store:
            try:
                if hasattr(self.vector_store, 'index') and self.vector_store.index:
                    info["document_count"] = self.vector_store.index.ntotal
                else:
                    info["document_count"] = 0
            except Exception as e:
                self.logger.warning(f"Error getting document count: {e}")
                info["document_count"] = "unknown"
        else:
            info["document_count"] = 0
            
        return info

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "content_types": {},
            "languages": {},
            "cache_size": 0,
            "chat_history_length": len(self.chat_history) // 2
        }

        # Get vector store stats if available
        if self.vector_store:
            # FAISS doesn't have direct access to document count and metadata
            # We'll estimate based on the index
            try:
                if hasattr(self.vector_store, 'index') and self.vector_store.index:
                    stats["total_documents"] = self.vector_store.index.ntotal
                
                # If we have the docstore, get more detailed stats
                if hasattr(self.vector_store, 'docstore') and self.vector_store.docstore:
                    for doc_id in self.vector_store.docstore._dict:
                        doc = self.vector_store.docstore._dict[doc_id]
                        if hasattr(doc, 'metadata'):
                            # Count content types
                            content_type = doc.metadata.get("content_type", "unknown")
                            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1

                            # Count languages
                            language = doc.metadata.get("language", "unknown")
                            stats["languages"][language] = stats["languages"].get(language, 0) + 1

                            # Estimate tokens (rough approximation)
                            stats["total_tokens"] += len(doc.page_content.split())
            except Exception as e:
                self.logger.warning(f"Error getting FAISS stats: {e}")

        # Get cache size
        cache_size = 0
        for root, _, files in os.walk(self.cache_dir):
            for file in files:
                cache_size += os.path.getsize(os.path.join(root, file))
        stats["cache_size"] = cache_size

        return stats

    def optimize(self) -> None:
        """Optimize the RAG system for better performance."""
        if not self.vector_store:
            return

        try:
            self.logger.info("Starting RAG system optimization")

            # Get current stats
            before_stats = self.get_stats()

            # Clear unnecessary cache entries
            self._clean_cache()

            # Get stats after optimization
            after_stats = self.get_stats()

            # Log optimization results
            self.logger.info(f"Optimization completed. Cache size reduced from "
                           f"{before_stats['cache_size']} to {after_stats['cache_size']} bytes")

        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

    def _clean_cache(self) -> None:
        """Clean old or invalid cache entries."""
        try:
            # Get all cache files
            cache_files = os.listdir(self.cache_dir)

            # Remove files older than 7 days
            from datetime import datetime, timedelta
            max_age = timedelta(days=7)
            now = datetime.now()

            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))

                if now - file_modified > max_age:
                    os.remove(cache_path)

        except Exception as e:
            self.logger.error(f"Error cleaning cache: {str(e)}")
            raise

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "k_retrieval": self.k_retrieval,
            "cache_dir": self.cache_dir,
            "stats": self.get_stats()
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], google_api_key: str) -> 'RepoRAG':
        """Create a RepoRAG instance from configuration."""
        instance = cls(
            google_api_key=google_api_key,
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            k_retrieval=config.get("k_retrieval", 4),
            cache_dir=config.get("cache_dir", "./cache")
        )
        return instance

    def save_vector_store(self, path: str = "./faiss_index") -> None:
        """Save FAISS vector store to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(path)
                self.logger.info(f"Vector store saved to {path}")
            except Exception as e:
                self.logger.error(f"Error saving vector store: {e}")
                raise
        else:
            self.logger.warning("No vector store to save")

    def load_vector_store(self, path: str = "./faiss_index") -> None:
        """Load FAISS vector store from disk."""
        try:
            if os.path.exists(path):
                self.vector_store = FAISS.load_local(
                    path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._initialize_qa_chain()
                self.logger.info(f"Vector store loaded from {path}")
            else:
                self.logger.warning(f"Path {path} does not exist")
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise

    def reset_system(self) -> None:
        """Completely reset the RAG system to initial state."""
        try:
            self.logger.info("Resetting RAG system to initial state")
            
            # Clear vector store
            self.clear_vector_store()
            
            # Clear QA chain
            if hasattr(self, 'qa_chain'):
                self.qa_chain = None
            
            # Clear chat history
            self.clear_memory()
            
            # Clear cache
            self.clear_cache()
            
            # Remove any saved FAISS indices
            faiss_paths = ["./faiss_index", "./repo_faiss_db"]
            for path in faiss_paths:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                    self.logger.info(f"Removed FAISS index at {path}")
            
            self.logger.info("RAG system reset completed")
            
        except Exception as e:
            self.logger.error(f"Error resetting system: {e}")
            raise

    def is_ready_for_queries(self) -> bool:
        """Check if the system is ready to answer queries."""
        return (
            hasattr(self, 'vector_store') and 
            self.vector_store is not None and
            hasattr(self, 'qa_chain') and 
            self.qa_chain is not None
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        status = {
            "ready_for_queries": self.is_ready_for_queries(),
            "vector_store_loaded": self.vector_store is not None,
            "qa_chain_initialized": hasattr(self, 'qa_chain') and self.qa_chain is not None,
            "repository_info": self.get_repository_info(),
            "system_stats": self.get_stats()
        }
        
        return status

# Helper functions for external use
def create_repo_rag(google_api_key: str, **kwargs) -> RepoRAG:
    """Create a new RepoRAG instance with given configuration."""
    return RepoRAG(google_api_key=google_api_key, **kwargs)

def load_repo_rag(config_path: str, google_api_key: str) -> RepoRAG:
    """Load a RepoRAG instance from a configuration file."""
    with open(config_path, 'r') as f:
        config = pickle.load(f)
    return RepoRAG.from_config(config, google_api_key)

def switch_to_new_repository(rag_instance: RepoRAG, content: str, clear_history: bool = True) -> None:
    """Convenience function to switch to a new repository."""
    rag_instance.switch_repository(content, clear_history=clear_history)

def reset_rag_system(rag_instance: RepoRAG) -> None:
    """Convenience function to completely reset the RAG system."""
    rag_instance.reset_system()