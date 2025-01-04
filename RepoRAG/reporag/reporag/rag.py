import logging
from typing import List, Dict, Any
import shutil
import os
import hashlib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class RepoRAG:
    def __init__(self, groq_api_key: str, model_name: str = "mixtral-8x7b-32768",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 k_retrieval: int = 4, cache_dir: str = "./cache"):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.cache_dir = cache_dir
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

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def ingest_content(self, content: str) -> None:
        """Ingest repository content into the vector store."""
        logger.info("Ingesting new content into the vector store.")

        try:
            # Clear the existing vector store
            if self.vector_store:
                logger.info("Clearing existing vector store.")
                self.vector_store = None

            # Clean and preprocess the content
            cleaned_content = self.preprocess_content(content)

            # Split text into chunks
            chunks = self.text_splitter.create_documents([cleaned_content])

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
                verbose=False
            )
            logger.info("Content ingestion completed successfully.")

        except Exception as e:
            logger.error(f"Error during content ingestion: {str(e)}")
            raise

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
            # Check cache for the query
            cache_key = self._generate_cache_key(question)
            cache_path = os.path.join(self.cache_dir, cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    response = pickle.load(f)
                logger.info("Retrieved response from cache.")
                return response

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

            # Cache the response
            with open(cache_path, 'wb') as f:
                pickle.dump(response["answer"], f)

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
            shutil.rmtree('./repo_chroma_db', ignore_errors=True)

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        logger.info("Clearing the cache directory.")
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def validate_content(self, content: str) -> bool:
        """Validate the content before ingestion."""
        if not content or not isinstance(content, str):
            logger.warning("Invalid content provided.")
            return False
        return True

    def preprocess_content(self, content: str) -> str:
        """Preprocess the content to improve quality."""
        # Lowercase the content
        content = content.lower()

        # Remove special characters and numbers
        content = re.sub(r'[^a-zA-Z\s]', '', content)

        # Tokenize the content into sentences
        sentences = sent_tokenize(content)

        # Tokenize sentences into words and remove stop words
        stop_words = set(stopwords.words('english'))
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [word for word in words if word not in stop_words]
            words = [self.lemmatizer.lemmatize(word) for word in words]
            cleaned_sentences.append(' '.join(words))

        # Join sentences back into a single string
        cleaned_content = ' '.join(cleaned_sentences)

        return cleaned_content

    def ingest_content_batch(self, contents: List[str]) -> None:
        """Ingest a batch of contents into the vector store."""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ingest_content, content) for content in contents if self.validate_content(content)]
            for future in futures:
                future.result()  # Ensure all tasks are completed

    def _generate_cache_key(self, question: str) -> str:
        """Generate a cache key for the query."""
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def dynamic_model_selection(self, question: str) -> None:
        """Dynamically select the model based on the query."""
        # Example logic for dynamic model selection
        if "specific_domain" in question:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name="specific_domain_model"
            )
        else:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name=self.model_name
            )