import os
from typing import Any
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

def get_setting(key: str, default: Any = None) -> Any:
    """Retrieve setting from os.environ or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# Sync Streamlit secrets into os.environ if running inside Streamlit
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
except Exception:
    pass

class Config:
    """Centralized configuration and singleton clients for the Hybrid GraphRAG Engine."""
    
    # -------------------------------------------------------------------------
    # API Keys & URIs (evaluated with fallback to Streamlit secrets)
    # -------------------------------------------------------------------------
    GROQ_API_KEY = get_setting("GROQ_API_KEY")
    PINECONE_API_KEY = get_setting("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = get_setting("PINECONE_INDEX_NAME", "hybrid-graphrag")
    
    NEO4J_URI = get_setting("NEO4J_URI")
    NEO4J_USERNAME = get_setting("NEO4J_USERNAME")
    NEO4J_PASSWORD = get_setting("NEO4J_PASSWORD")
    
    # -------------------------------------------------------------------------
    # Application Constants
    # -------------------------------------------------------------------------
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Groq API limits - Free Tier
    GROQ_RPM_LIMIT = 30
    # Add a slight buffer to the interval
    GROQ_REQUEST_INTERVAL_SECONDS = (60.0 / GROQ_RPM_LIMIT) + 0.5 

    _groq_client = None
    _pinecone_client = None
    _neo4j_driver = None

    @classmethod
    def refresh_from_secrets(cls):
        """Re-read credentials dynamically in case secrets were added after module import."""
        cls.GROQ_API_KEY = get_setting("GROQ_API_KEY") or cls.GROQ_API_KEY
        cls.PINECONE_API_KEY = get_setting("PINECONE_API_KEY") or cls.PINECONE_API_KEY
        cls.PINECONE_INDEX_NAME = get_setting("PINECONE_INDEX_NAME", "hybrid-graphrag")
        cls.NEO4J_URI = get_setting("NEO4J_URI") or cls.NEO4J_URI
        cls.NEO4J_USERNAME = get_setting("NEO4J_USERNAME") or cls.NEO4J_USERNAME
        cls.NEO4J_PASSWORD = get_setting("NEO4J_PASSWORD") or cls.NEO4J_PASSWORD

    @classmethod
    def get_groq_client(cls) -> Groq:
        cls.refresh_from_secrets()
        if cls._groq_client is None:
            if not cls.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in environment variables or Streamlit secrets.")
            cls._groq_client = Groq(api_key=cls.GROQ_API_KEY)
        return cls._groq_client

    @classmethod
    def get_pinecone_client(cls) -> Pinecone:
        cls.refresh_from_secrets()
        if cls._pinecone_client is None:
            if not cls.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is not set in environment variables or Streamlit secrets.")
            cls._pinecone_client = Pinecone(api_key=cls.PINECONE_API_KEY)
        return cls._pinecone_client

    @classmethod
    def get_neo4j_driver(cls):
        cls.refresh_from_secrets()
        if cls._neo4j_driver is None:
            if not all([cls.NEO4J_URI, cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD]):
                missing = [name for name, val in [("NEO4J_URI", cls.NEO4J_URI), ("NEO4J_USERNAME", cls.NEO4J_USERNAME), ("NEO4J_PASSWORD", cls.NEO4J_PASSWORD)] if not val]
                raise ValueError(f"Neo4j credentials incomplete. Missing: {', '.join(missing)}")
            cls._neo4j_driver = GraphDatabase.driver(
                cls.NEO4J_URI, auth=(cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD)
            )
        return cls._neo4j_driver

    @classmethod
    def close_connections(cls):
        """Safely close open database connections."""
        if cls._neo4j_driver is not None:
            cls._neo4j_driver.close()
            cls._neo4j_driver = None
