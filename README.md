# Hybrid GraphRAG Engine

A Hybrid Graph-based Retrieval-Augmented Generation (GraphRAG) application combining semantic vector search (Pinecone), structural knowledge graphs (Neo4j), and high-speed LLM inference (Groq).

---

## Architecture Overview

1. **Semantic Search**: Text embeddings generated with `sentence-transformers/all-MiniLM-L6-v2` and stored in **Pinecone**.
2. **Graph Knowledge Base**: Extracted entities and relationships stored in **Neo4j**.
3. **Synthesis Engine**: LLaMA-3.1 powered by **Groq** to synthesize contextual answers from both semantic and relational graph context.
4. **User Interface**: Interactive **Streamlit** dashboard for PDF ingestion and real-time question answering.

---

## Environment Configuration

Create a `.env` file in the root or in `graphrag-project/` (or configure secrets in your cloud provider):

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hybrid-graphrag
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

## Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run graphrag-project/app.py
   ```

---

## Cloud Deployment

### Option 1: Streamlit Community Cloud (Recommended - Free & 2-Minute Setup)

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and choose this repository:
   - **Branch**: `main`
   - **Main file path**: `graphrag-project/app.py`
4. Expand **Advanced settings** -> **Secrets** and paste your environment variables in TOML format:
   ```toml
   GROQ_API_KEY = "your_key"
   PINECONE_API_KEY = "your_key"
   PINECONE_INDEX_NAME = "hybrid-graphrag"
   NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "your_password"
   ```
5. Click **Deploy**.

---

### Option 2: Render / Railway (Docker Deployment)

1. Connect your repository to [Render](https://render.com/) or [Railway](https://railway.app/).
2. Select **Web Service** using Docker (the root `Dockerfile` is pre-configured).
3. Set your environment variables in the provider dashboard (`GROQ_API_KEY`, `PINECONE_API_KEY`, etc.).
4. The service will build and run on port `8501`.

