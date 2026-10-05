# Strict RAG-based PDF Q&A Application (Groq Cloud + HuggingFace)

A modular, zero-hallucination document question-answering web application built with **Streamlit**, **LangChain**, **Groq Cloud (LPU Inference)**, **HuggingFace Embeddings**, and **ChromaDB**.

---

## 🎯 Key Architectural Features

- **Strict Prompt Constraint:** Instructs the LLM to answer **solely** based on retrieved document chunks. If the answer cannot be found directly in the uploaded PDF, it strictly replies:
  > `"Information not found in the document."`
- **Zero Hallucination Determinism:** Uses `temperature=0.0` for deterministic generation.
- **Groq Cloud Speed:** Powered by `ChatGroq` running `llama-3.1-8b-instant` for ultra-fast, near-instantaneous token generation.
- **Free Local Embeddings:** Uses `langchain-huggingface` with `all-MiniLM-L6-v2` (`sentence-transformers`), running 100% locally on your machine with zero third-party embedding API costs.
- **Controlled Text Chunking:** `RecursiveCharacterTextSplitter` configured with a chunk size of `1000` and an overlap of `200`.
- **Local ChromaDB Vector Store:** Automatically embeds and indexes documents into local/in-memory Chroma collections.
- **Streaming UI:** Streams tokens directly to the Streamlit chat interface with expandable source chunk and page-number inspection.

---

## 📁 Project Structure

```
├── app.py                      # Streamlit chat interface & Groq configuration
├── core/
│   ├── __init__.py             # Core package initializer
│   ├── document_processor.py   # PyPDF extraction & recursive chunking (1000/200)
│   ├── vector_store.py         # HuggingFace (all-MiniLM-L6-v2) & ChromaDB storage
│   └── rag_chain.py            # LangChain LCEL pipeline with ChatGroq (temp=0.0)
├── requirements.txt            # Project dependencies
├── .env.example                # Template for environment variables (GROQ_API_KEY)
└── README.md                   # Documentation & setup instructions
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Add your Groq Cloud API key from [Groq Console](https://console.groq.com/):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Alternatively, you can input your key directly into the Streamlit sidebar at runtime).*

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```
