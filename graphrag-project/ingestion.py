import time
import json
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from config import Config

class IngestionPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_ID)
        self.groq_client = Config.get_groq_client()
        self.pinecone_index = Config.get_pinecone_client().Index(Config.PINECONE_INDEX_NAME)
        self.neo4j_driver = Config.get_neo4j_driver()

    def process_pdf(self, file_path: str) -> List[str]:
        """Extracts and chunks text from a PDF file."""
        reader = PdfReader(file_path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Basic chunking by paragraph/lines or just simple character split
                # Real implementation would use LangChain's RecursiveCharacterTextSplitter etc.
                words = text.split()
                chunk_size = 200
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)
        return chunks

    def embed_and_upsert_vectors(self, chunks: List[str]):
        """Embeds text chunks locally and upserts to Pinecone."""
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk).tolist()
            # In production, use more robust IDs
            chunk_id = f"chunk_{int(time.time())}_{i}"
            vectors.append({"id": chunk_id, "values": embedding, "metadata": {"text": chunk}})
            
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.pinecone_index.upsert(vectors=vectors[i:i + batch_size])
        
        return [v["id"] for v in vectors]

    def extract_graph_entities(self, chunk: str) -> Dict[str, Any]:
        """Extracts nodes and edges from a text chunk using Groq LLM with rate limiting."""
        prompt = f"""
        Analyze the following text and extract named entities and their relationships.
        Format the output as a JSON object with 'nodes' and 'edges'.
        Nodes should have 'id' and 'label'. Edges should have 'source', 'target', and 'type'.
        
        Text: {chunk}
        """
        
        # Enforce Rate Limiting
        time.sleep(Config.GROQ_REQUEST_INTERVAL_SECONDS)
        
        response = self.groq_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"nodes": [], "edges": []}

    def upsert_graph_to_neo4j(self, graph_data: Dict[str, Any], chunk_id: str):
        """Upserts nodes and edges into Neo4j."""
        query = """
        // Example Cypher query template to merge nodes and edges
        UNWIND $nodes AS node
        MERGE (n:Entity {id: node.id})
        SET n.label = node.label, n.source_chunk = $chunk_id
        
        WITH $edges AS edges
        UNWIND edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:RELATES_TO {type: edge.type}]->(target)
        """
        
        with self.neo4j_driver.session() as session:
            session.run(query, nodes=graph_data.get("nodes", []), 
                        edges=graph_data.get("edges", []), 
                        chunk_id=chunk_id)

    def run_pipeline(self, file_path: str):
        """Executes the full ingestion pipeline (A & B)."""
        print(f"Starting ingestion for {file_path}")
        chunks = self.process_pdf(file_path)
        print(f"Extracted {len(chunks)} chunks.")
        
        # Track A: Vector Indexing
        print("Embedding and upserting vectors to Pinecone...")
        chunk_ids = self.embed_and_upsert_vectors(chunks)
        
        # Track B: Graph Indexing
        print("Extracting graph entities and upserting to Neo4j...")
        for i, chunk in enumerate(chunks):
            graph_data = self.extract_graph_entities(chunk)
            self.upsert_graph_to_neo4j(graph_data, chunk_ids[i])
            print(f"Processed chunk {i+1}/{len(chunks)} for graph indexing.")
            
        print("Ingestion complete!")

if __name__ == "__main__":
    # Example usage
    # pipeline = IngestionPipeline()
    # pipeline.run_pipeline("sample.pdf")
    pass
