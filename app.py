"""Streamlit Web Application: Strict RAG-based PDF Q&A Chatbot powered by Groq Cloud & HuggingFace."""

import os
import streamlit as st
from dotenv import load_dotenv

from core.document_processor import load_and_split_pdf
from core.vector_store import create_vector_store, get_embedding_model
from core.rag_chain import build_rag_chain, execute_strict_rag_stream

# Load environment variables from .env file if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Strict RAG - PDF Document Q&A (Groq Cloud)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
    .badge-ready {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-waiting {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


# Cached warm-up of embedding model
@st.cache_resource(show_spinner=False)
def load_cached_embeddings():
    return get_embedding_model("all-MiniLM-L6-v2")


# Pre-warm embeddings in background
load_cached_embeddings()


# ==============================================================================
# SIDEBAR: Configuration, Credentials, and Upload
# ==============================================================================
with st.sidebar:
    st.title("⚡ Groq Engine Control")
    st.markdown("Ultra-fast inference with **Groq Cloud** & local **HuggingFace** embeddings.")
    
    # 1. Groq API Key Input
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_api_key,
        type="password",
        help="Enter your gsk_... key from console.groq.com. Can also be set in .env."
    )

    # 2. Model Selection (Groq supported models)
    model_choice = st.selectbox(
        "Groq LLM Engine",
        options=[
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0,
        help="Deterministic temperature is locked at 0.0 for zero hallucination."
    )

    # 3. Retrieval Parameters
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of context segments fetched from local ChromaDB."
    )

    st.caption("📦 Embeddings: `all-MiniLM-L6-v2` (Local HuggingFace, no extra API fees)")
    st.divider()

    # 4. PDF File Uploader
    st.subheader("📁 Document Ingestion")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload the PDF to extract, chunk, and index."
    )

    # Detect if user uploaded a different document
    if uploaded_pdf is not None and st.session_state.current_file_name != uploaded_pdf.name:
        st.session_state.document_processed = False
        st.session_state.retriever = None
        st.session_state.chain = None

    # Process Document Button
    if uploaded_pdf is not None:
        process_button = st.button("🚀 Process & Index PDF", use_container_width=True, type="primary")

        # Explicit button trigger to prevent duplicate concurrent runs
        if process_button:
            if not api_key:
                st.error("Please provide a Groq API Key before proceeding.")
            else:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    with st.spinner("Processing PDF (extracting text, chunking, and embedding)..."):
                        try:
                            # 1. Chunking
                            chunks = load_and_split_pdf(
                                file_bytes_or_path=uploaded_pdf,
                                file_name=uploaded_pdf.name,
                                chunk_size=1000,
                                chunk_overlap=200
                            )

                            # 2. Vector DB Indexing with cached HuggingFace embeddings
                            vector_store = create_vector_store(
                                documents=chunks
                            )

                            # 3. Build Strict RAG Chain with Groq
                            retriever, chain = build_rag_chain(
                                vector_store=vector_store,
                                api_key=api_key,
                                model_name=model_choice,
                                top_k=top_k
                            )

                            st.session_state.chunk_count = len(chunks)
                            st.session_state.retriever = retriever
                            st.session_state.chain = chain
                            st.session_state.document_processed = True
                            st.session_state.current_file_name = uploaded_pdf.name
                            st.session_state.messages = []  # Reset chat on new document

                            st.success(f"Indexed {len(chunks)} chunks successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error during ingestion: {str(e)}")

    st.divider()

    # Document Status Indicator
    if st.session_state.document_processed:
        st.markdown(
            f'<div class="status-badge badge-ready">🟢 Ready: {st.session_state.current_file_name} ({st.session_state.chunk_count} chunks)</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge badge-waiting">🟡 Waiting for Document Upload</div>', 
            unsafe_allow_html=True
        )

    # Reset Chat History Button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">Strict RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Zero-hallucination Q&A powered by <b>Groq Cloud</b> & <b>HuggingFace</b>. '
    'If the answer is not present in the document, the model will output: <i>"Information not found in the document."</i></div>',
    unsafe_allow_html=True
)

# Display existing conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, src in enumerate(msg["sources"]):
                    page = src.get("page", "Unknown")
                    st.markdown(f"**Chunk {idx + 1} (Page {page})**")
                    st.text(src.get("content", ""))

# Chat Input & Strict Generation Logic
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    # Guard 1: Verify API Key
    if not api_key:
        st.warning("⚠️ Please provide a Groq API Key in the left sidebar to proceed.")
        st.stop()

    # Guard 2: Verify Document Processed
    if not st.session_state.document_processed or st.session_state.chain is None:
        st.warning("⚠️ Please upload and process a PDF document first using the sidebar button.")
        st.stop()

    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Execute RAG Retrieval & Streaming Generation
    with st.chat_message("assistant"):
        try:
            stream_gen, retrieved_docs = execute_strict_rag_stream(
                query=prompt,
                retriever=st.session_state.retriever,
                chain=st.session_state.chain
            )

            # Stream the generated tokens into the UI
            response_text = st.write_stream(stream_gen)

            # Package source chunks for display and history persistence
            source_data = []
            for doc in retrieved_docs:
                page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
                source_data.append({
                    "page": page,
                    "content": doc.page_content
                })

            # Show retrieved context in an expander for full auditability
            if source_data:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, src in enumerate(source_data):
                        st.markdown(f"**Chunk {idx + 1} (Page {src['page']})**")
                        st.text(src["content"])

            # Save assistant response to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": source_data
            })

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
