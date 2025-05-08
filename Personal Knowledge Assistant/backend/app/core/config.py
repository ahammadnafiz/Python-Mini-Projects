# app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Personal Knowledge Assistant"
    API_PREFIX: str = "/api"
    GROQ_API_KEY_MODEL: str = os.getenv("GROQ_API_KEY_MODEL", "")  
    # Vector store settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    # Embedding settings
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL: str = "intfloat/e5-small"
    # EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # LLM settings
    LLM_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # RAG settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    TOP_K_RESULTS: int = 5

settings = Settings()