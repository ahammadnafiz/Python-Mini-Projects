"""
MedBot: A medical chatbot system using document retrieval and LLM-based question answering.

This module implements a document retrieval and question answering system specialized for medical
information. It loads PDF documents, processes them, creates embeddings, and stores them in
a vector database for efficient retrieval.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class ConfigManager:
    """Manages configuration and environment variables."""
    
    def __init__(self):
        """Initialize configuration manager and load environment variables."""
        # Change to parent directory for correct .env file location
        original_dir = os.getcwd()
        os.chdir('../')
        
        # Load environment variables
        load_dotenv()
        
        # Store API keys
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        
        # Set environment variables for dependent libraries
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        os.environ["GROQ_API_KEY"] = self.groq_api_key
        
        # Return to original directory
        os.chdir(original_dir)


class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 20):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf_documents(self, directory_path: str) -> List:
        """
        Load PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(
            directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)


class EmbeddingManager:
    """Handles document embeddings creation and storage."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embeddings = None
    
    def load_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Load the embedding model.
        
        Returns:
            HuggingFaceEmbeddings object
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self.embeddings


class PineconeManager:
    """Manages Pinecone vector database operations."""
    
    def __init__(self, api_key: str, index_name: str = "medibot"):
        """
        Initialize Pinecone manager.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=api_key)
        self.vector_store = None
    
    def create_index(self, dimension: int = 384, metric: str = "cosine", 
                     cloud: str = "aws", region: str = "us-east-1") -> None:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Dimension of the embedding vectors
            metric: Distance metric to use for similarity
            cloud: Cloud provider to use
            region: Cloud region to use
        """
        self.pinecone_client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
    
    def upload_documents(self, documents: List, embeddings) -> PineconeVectorStore:
        """
        Upload documents to the Pinecone index.
        
        Args:
            documents: List of document chunks to upload
            embeddings: Embeddings model to use
            
        Returns:
            PineconeVectorStore object
        """
        self.vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            index_name=self.index_name,
            embedding=embeddings
        )
        return self.vector_store
    
    def load_existing_index(self, embeddings) -> PineconeVectorStore:
        """
        Load an existing Pinecone index.
        
        Args:
            embeddings: Embeddings model to use
            
        Returns:
            PineconeVectorStore object
        """
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings
        )
        return self.vector_store
    
    def create_retriever(self, search_type: str = "similarity", k: int = 3):
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: Type of search to perform
            k: Number of results to return
            
        Returns:
            Retriever object
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create an index first.")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )


class DocChat:
    """Handles document-based chat interactions."""
    
    def __init__(self, groq_api_key: str, retriever, model_name: str = "mixtral-8x7b-32768"):
        """
        Initialize DocChat with document retriever and model parameters.
        
        Args:
            groq_api_key: API key for Groq
            retriever: Document retriever for fetching relevant content
            model_name: Model name to use
        """
        self.retriever = retriever
        self.llm = None
        self.qa_chain = None
        self.chat_history = []
        self.prompts = self._create_prompts()
        self.initialize_groq_components(groq_api_key, model_name)

    def _create_prompts(self) -> Dict:
        """
        Create prompt templates for different query types.
        
        Returns:
            Dictionary of prompt templates
        """
        # Base system message template for all queries
        base_system_template = """You are MedBot, an advanced AI assistant specialized in medical knowledge.

CAPABILITIES:
- Provide accurate, evidence-based medical information
- Explain medical terminology in clear, accessible language
- Interpret symptoms and medical conditions with precision
- Reference medical literature and guidelines when appropriate

CONSTRAINTS:
- You are NOT a replacement for professional medical diagnosis or treatment
- Always clarify that users should consult healthcare providers for personal medical advice
- Maintain strict medical accuracy - if unsure, acknowledge limitations
- Avoid making definitive diagnostic statements

RESPONSE FORMAT:
- Begin with a clear, direct answer to the question
- Provide context and additional relevant information
- Use bullet points for symptoms, treatments, or key facts when appropriate
- Include brief mention of relevant medical guidelines or consensus when applicable
- End with a reminder about consulting healthcare professionals when appropriate

DOCUMENT CONTEXT BELOW:
{context}

Remember: Base your responses on the document context provided. If the context doesn't contain relevant information, acknowledge this limitation.
"""

        # Human template focusing on the question
        human_template = """
{input}
"""

        # Create a ChatPromptTemplate
        base_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(base_system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Create specialized templates for different query types
        prompts = {
            "general": base_prompt,
            
            "medical": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
ADDITIONAL MEDICAL INSTRUCTIONS:
- Use proper medical terminology with layperson explanations
- Include ICD codes when identifying specific conditions
- Reference standard treatment protocols when applicable
- Mention both conventional and evidence-based alternative approaches
- Clarify levels of evidence (e.g., RCT, meta-analysis, case studies)
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ]),
            
            "educational": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
EDUCATIONAL FOCUS:
- Structure responses with clear learning objectives
- Define all medical terms when first introduced
- Build explanations from basic concepts to more complex ideas
- Use anatomical references and physiological processes to explain mechanisms
- Incorporate mnemonics or memory aids when helpful
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ]),
            
            "detailed": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
DETAILED ANALYSIS REQUIREMENTS:
- Provide in-depth coverage of the topic with subsections
- Include epidemiological data when relevant
- Discuss pathophysiology in detail
- Cover differential diagnosis considerations
- Elaborate on diagnostic criteria and testing modalities
- Detail treatment approaches with medication classes/options
- Address prognosis and complications
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        }
        
        return prompts

    def initialize_groq_components(self, groq_api_key: str, model_name: str) -> None:
        """
        Initialize Groq chat components.
        
        Args:
            groq_api_key: API key for Groq
            model_name: Model name to use
        """
        try:
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model_name=model_name,
                temperature=0.2,  # Lower temperature for more consistent medical responses
                max_tokens=2048    # Ensure sufficient tokens for detailed answers
            )
            self._initialize_qa_chain()
        except Exception as e:
            raise Exception(f"Failed to initialize Groq components: {str(e)}")

    def _initialize_qa_chain(self) -> None:
        """Initialize document-based retrieval and answering chain."""
        try:
            # Create the question answering chain with the general prompt by default
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                self.prompts["general"],
                document_variable_name="context"
            )
            
            # Create the retrieval chain
            self.qa_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        except Exception as e:
            raise Exception(f"Failed to initialize QA chain: {str(e)}")

    def query(self, question: str, query_type: str = "general") -> Dict:
        """
        Execute chat query.
        
        Args:
            question: User question
            query_type: Type of query (medical, educational, detailed, or general)
            
        Returns:
            Dict: Response from the QA chain
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Make sure to initialize components first.")
        
        # Verify the query type exists or default to general
        if query_type not in self.prompts:
            print(f"Warning: Query type '{query_type}' not found. Using 'general' instead.")
            query_type = "general"
        
        try:
            # Create a specialized chain for this query type
            specialized_qa_chain = create_stuff_documents_chain(
                self.llm, 
                self.prompts[query_type],
                document_variable_name="context"
            )
            
            specialized_retrieval_chain = create_retrieval_chain(
                self.retriever, specialized_qa_chain
            )
            
            # Execute the query
            response = specialized_retrieval_chain.invoke({
                "input": question,
                "chat_history": self.get_chat_history()
            })
            
            # Add metadata to the response
            response["query_type"] = query_type
            response["timestamp"] = datetime.now().isoformat()
            
            # Update chat history
            self._update_chat_history(question, response["answer"])
            return response
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            return {"answer": error_msg, "error": True, "query_type": query_type}

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """
        Get formatted chat history for LangChain.
        
        Returns:
            List of (human_message, ai_message) tuples
        """
        formatted_history = []
        
        # Create pairs of human and AI messages
        for i in range(0, len(self.chat_history) - 1, 2):
            if i + 1 < len(self.chat_history):
                if isinstance(self.chat_history[i], HumanMessage) and isinstance(self.chat_history[i+1], AIMessage):
                    formatted_history.append((
                        self.chat_history[i].content, 
                        self.chat_history[i+1].content
                    ))
                
        return formatted_history

    def _update_chat_history(self, question: str, answer: str) -> None:
        """
        Update conversation history.
        
        Args:
            question: Human question
            answer: AI answer
        """
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

    def reset_chat_history(self) -> None:
        """Reset the chat history."""
        self.chat_history = []

    def add_context(self, context: str, context_type: str = "system") -> None:
        """
        Add additional context to the DocChat.
        
        Args:
            context: Additional context to consider
            context_type: Type of context (system, human, ai)
        """
        if context_type == "system":
            self.chat_history.append(SystemMessage(content=context))
        elif context_type == "human":
            self.chat_history.append(HumanMessage(content=context))
        elif context_type == "ai":
            self.chat_history.append(AIMessage(content=context))
        else:
            raise ValueError(f"Invalid context type: {context_type}. Use 'system', 'human', or 'ai'.")


class MedBotApp:
    """Main application class for MedBot."""
    
    def __init__(self, data_dir: str = "Data/"):
        """
        Initialize MedBot application.
        
        Args:
            data_dir: Directory containing medical PDF documents
        """
        self.data_dir = data_dir
        self.config = ConfigManager()
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.pinecone_manager = None
        self.doc_chat = None
    
    def setup(self, model_name, create_new_index: bool = False) -> None:
        """
        Set up the MedBot application.
        
        Args:
            create_new_index: Whether to create a new Pinecone index
        """
        # Load and process documents
        documents = self.doc_processor.load_pdf_documents(self.data_dir)
        text_chunks = self.doc_processor.split_documents(documents)
        print(f"Processed {len(text_chunks)} text chunks from documents")
        
        # Load embeddings
        embeddings = self.embedding_manager.load_embeddings()
        
        # Set up Pinecone
        self.pinecone_manager = PineconeManager(self.config.pinecone_api_key)
        
        if create_new_index:
            # Create a new index and upload documents
            self.pinecone_manager.create_index()
            self.pinecone_manager.upload_documents(text_chunks, embeddings)
        else:
            # Load existing index
            self.pinecone_manager.load_existing_index(embeddings)
        
        # Create retriever
        retriever = self.pinecone_manager.create_retriever()
        
        # Set up DocChat
        self.doc_chat = self.create_doc_chat(retriever, model_name)
    
    def create_doc_chat(self, retriever, model_name: str = "mixtral-8x7b-32768") -> Optional[DocChat]:
        """
        Create and return a DocChat instance with error handling.
        
        Args:
            retriever: Document retriever
            model_name: Model name to use
            
        Returns:
            DocChat instance or None if initialization fails
        """
        try:
            doc_chat = DocChat(
                groq_api_key=self.config.groq_api_key,
                retriever=retriever,
                model_name=model_name
            )
            return doc_chat
        except Exception as e:
            print(f"Failed to create DocChat: {str(e)}")
            return None
    
    def query(self, question: str, query_type: str = "medical") -> Dict:
        """
        Make a query to the DocChat.
        
        Args:
            question: Question to ask
            query_type: Type of query (medical, educational, detailed, or general)
            
        Returns:
            Response from the DocChat
        """
        if not self.doc_chat:
            raise ValueError("DocChat not initialized. Set up the application first.")
        
        return self.doc_chat.query(question, query_type)