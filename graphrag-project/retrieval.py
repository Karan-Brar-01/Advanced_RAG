import time
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import Config

class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_ID)
        self.groq_client = Config.get_groq_client()
        self.pinecone_index = Config.get_pinecone_client().Index(Config.PINECONE_INDEX_NAME)
        self.neo4j_driver = Config.get_neo4j_driver()

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieves top-k semantically similar text chunks from Pinecone."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        context_chunks = [match["metadata"]["text"] for match in results.get("matches", [])]
        return context_chunks

    def extract_query_entities(self, query: str) -> List[str]:
        """Extracts entities from the user query using Groq LLM to use as graph entry points."""
        prompt = f"""
        Extract the core named entities from the following query. 
        Return ONLY a JSON object with a single key 'entities' containing an array of strings representing the entity names.
        
        Query: {query}
        """
        
        # Enforce Rate Limiting
        time.sleep(Config.GROQ_REQUEST_INTERVAL_SECONDS)
        
        response = self.groq_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            data = json.loads(response.choices[0].message.content)
            # Depending on how the LLM formats it, extract the list
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
            return data if isinstance(data, list) else []
        except:
            return []

    def graph_traversal_search(self, entities: List[str]) -> str:
        """Queries Neo4j using extracted entities to fetch sub-graph relational context."""
        if not entities:
            return ""
            
        query = """
        MATCH (n:Entity)-[r]->(m)
        WHERE n.id IN $entities OR n.label IN $entities
        RETURN n.id, type(r), m.id
        LIMIT 50
        """
        
        subgraph_context = []
        with self.neo4j_driver.session() as session:
            result = session.run(query, entities=entities)
            for record in result:
                subgraph_context.append(f"{record['n.id']} --[{record['type(r)']}]--> {record['m.id']}")
                
        return "\n".join(subgraph_context)

    def retrieve_context(self, query: str) -> Dict[str, Any]:
        """Executes the hybrid retrieval logic unifying vector and graph context."""
        
        # Track A: Semantic Search
        semantic_context = self.semantic_search(query)
        
        # Track B: Graph Traversal
        query_entities = self.extract_query_entities(query)
        graph_context = self.graph_traversal_search(query_entities)
        
        return {
            "semantic_context": semantic_context,
            "graph_context": graph_context,
            "query_entities": query_entities
        }
