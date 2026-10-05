"""Vector store module: Initializes HuggingFace embeddings and manages local ChromaDB storage."""

import os
import uuid
from typing import List, Optional
import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Global cache for embeddings instance to prevent repeated model reloading
_EMBEDDINGS_INSTANCE = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embeddings model (local inference).
    Caches the instance in memory to prevent repeated weight loads.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_INSTANCE


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Chroma:
    """
    Creates a new Chroma vector store from a list of chunked documents.
    Uses an ephemeral in-memory client by default with unique collection names
    to prevent cross-session collisions.
    
    Args:
        documents: List of chunked Document objects.
        persist_directory: Path to store local ChromaDB files (optional).
        collection_name: ChromaDB collection name (optional, defaults to unique hash).
        embedding_model_name: HuggingFace model identifier.
        
    Returns:
        Chroma: Initialized Chroma vector store instance.
    """
    embeddings = get_embedding_model(model_name=embedding_model_name)
    coll_name = collection_name or f"pdf_rag_{uuid.uuid4().hex[:8]}"

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=coll_name
        )
    else:
        # Isolated ephemeral in-memory client
        client = chromadb.Client()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            collection_name=coll_name
        )

    return vector_store


def get_existing_vector_store(
    persist_directory: str,
    collection_name: str = "pdf_rag_collection",
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Chroma:
    """
    Loads an existing local ChromaDB vector store using HuggingFaceEmbeddings.
    """
    embeddings = get_embedding_model(model_name=embedding_model_name)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
