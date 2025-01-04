# repoprompter/repoprompter/rag.py

import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepoRAG:
    def __init__(self, groq_api_key: str, model_name: str = "mixtral-8x7b-32768",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 k_retrieval: int = 4):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.chat_history = []  # Initialize as empty list

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name
        )

        # Initialize vector store
        self.vector_store = None

        # Initialize prompt template
        self.system_template = """You are an AI assistant specialized in understanding and explaining repository content.
        Use the following pieces of context to answer questions about the repository:

        {context}

        If you don't know something or can't find it in the context, say so."""

    def ingest_content(self, content: str) -> None:
        """Ingest repository content into the vector store."""
        logger.info("Ingesting new content into the vector store.")

        # Clear the existing vector store
        if self.vector_store:
            logger.info("Clearing existing vector store.")
            self.vector_store = None

        # Split text into chunks
        chunks = self.text_splitter.create_documents([content])

        # Create or update vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./repo_chroma_db"
        )

        # Initialize retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_retrieval}
            ),
            return_source_documents=True,
            verbose=True
        )
        logger.info("Content ingestion completed successfully.")

    def get_chat_history(self) -> List:
        """Convert chat history to the format expected by the chain."""
        formatted_history = []
        for i in range(0, len(self.chat_history), 2):
            if i + 1 < len(self.chat_history):
                formatted_history.append(
                    (self.chat_history[i].content, self.chat_history[i + 1].content)
                )
        return formatted_history

    def query(self, question: str) -> str:
        """Query the RAG system about repository content."""
        if self.vector_store is None:
            raise ValueError("No content has been ingested yet. Please call ingest_content first.")

        try:
            # Get properly formatted chat history
            formatted_history = self.get_chat_history()

            # Execute the chain
            response = self.qa_chain({
                "question": question,
                "chat_history": formatted_history
            })

            # Update chat history with the new interaction
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response["answer"]))

            return response["answer"]

        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            raise

    def clear_memory(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        logger.info("Chat history cleared.")

    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        if self.vector_store:
            logger.info("Clearing the vector store.")
            self.vector_store = None
            # Optionally delete the persisted vector store
            # import shutil
            # shutil.rmtree('./repo_chroma_db', ignore_errors=True)