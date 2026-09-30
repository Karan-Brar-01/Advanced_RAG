import os
import tempfile
import streamlit as st
from config import Config
from retrieval import HybridRetriever
from ingestion import IngestionPipeline

def synthesize_answer(query: str, context: dict) -> str:
    """Uses Groq LLM to synthesize a final answer based on the hybrid context."""
    groq_client = Config.get_groq_client()
    
    semantic_text = "\n\n".join(context.get("semantic_context", []))
    graph_text = context.get("graph_context", "")
    
    prompt = f"""
    You are an expert assistant answering questions based on the provided context.
    
    You have two types of context available:
    1. Semantic Text Context (Exact text chunks retrieved from documents)
    2. Structural Graph Context (Relational data showing how entities are connected)
    
    Use BOTH contexts to synthesize a comprehensive, accurate, and grounded answer.
    If the context does not contain the answer, state that you do not know.
    
    ====================
    SEMANTIC TEXT CONTEXT:
    {semantic_text}
    
    ====================
    STRUCTURAL GRAPH CONTEXT:
    {graph_text}
    
    ====================
    USER QUERY: {query}
    
    ANSWER:
    """
    
    response = groq_client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Hybrid GraphRAG Engine", layout="wide")
    st.title("Hybrid GraphRAG Search")
    st.markdown("Powered by Groq, Pinecone, Neo4j, and local Sentence-Transformers.")
    
    # Sidebar for Document Ingestion
    st.sidebar.header("Document Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF to index", type=["pdf"])
    
    if uploaded_file is not None:
        if st.sidebar.button("Process Document"):
            with st.sidebar.status("Processing Document..."):
                try:
                    # Save uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    st.write("Initializing ingestion pipeline...")
                    pipeline = IngestionPipeline()
                    st.write("Running ingestion pipeline (check console for detailed progress)...")
                    pipeline.run_pipeline(tmp_file_path)
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    
                    st.success("Document ingested successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
    
    # Initialize Retriever
    @st.cache_resource
    def get_retriever():
        return HybridRetriever()
        
    try:
        retriever = get_retriever()
    except Exception as e:
        st.error(f"Failed to initialize retriever. Check your .env configuration. Error: {e}")
        return

    # User Input
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        with st.spinner("Executing Hybrid Retrieval Pipeline..."):
            try:
                # 1. Retrieve Context
                context = retriever.retrieve_context(query)
                
                # Display intermediate state tracking
                with st.expander("Pipeline State Tracking: Extracted Query Entities"):
                    st.write(context["query_entities"])
                
                with st.expander("Pipeline State Tracking: Semantic Context (Pinecone)"):
                    for i, chunk in enumerate(context["semantic_context"]):
                        st.markdown(f"**Chunk {i+1}:** {chunk}")
                        
                with st.expander("Pipeline State Tracking: Graph Context (Neo4j)"):
                    st.code(context["graph_context"])
                
                # 2. Synthesize Answer
                st.subheader("Synthesized Answer")
                with st.spinner("Synthesizing answer with Groq LLM..."):
                    answer = synthesize_answer(query, context)
                    st.write(answer)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
