import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration and singleton clients for the Hybrid GraphRAG Engine."""
    
    # -------------------------------------------------------------------------
    # API Keys & URIs
    # -------------------------------------------------------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-graphrag")
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
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
    def get_groq_client(cls) -> Groq:
        if cls._groq_client is None:
            if not cls.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in environment variables.")
            cls._groq_client = Groq(api_key=cls.GROQ_API_KEY)
        return cls._groq_client

    @classmethod
    def get_pinecone_client(cls) -> Pinecone:
        if cls._pinecone_client is None:
            if not cls.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is not set in environment variables.")
            cls._pinecone_client = Pinecone(api_key=cls.PINECONE_API_KEY)
        return cls._pinecone_client

    @classmethod
    def get_neo4j_driver(cls):
        if cls._neo4j_driver is None:
            if not all([cls.NEO4J_URI, cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD]):
                raise ValueError("Neo4j credentials are not fully set in environment variables.")
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
