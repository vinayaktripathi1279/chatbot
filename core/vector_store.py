"""Vector store module: Initializes HuggingFace embeddings and manages local ChromaDB storage."""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embeddings model (local inference).
    Uses 'all-MiniLM-L6-v2' by default.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: str = "pdf_rag_collection",
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Chroma:
    """
    Creates a new Chroma vector store from a list of chunked documents.
    If persist_directory is specified, vectors will be saved locally.
    
    Args:
        documents: List of chunked Document objects.
        persist_directory: Path to store local ChromaDB files (optional).
        collection_name: ChromaDB collection name.
        embedding_model_name: HuggingFace model identifier.
        
    Returns:
        Chroma: Initialized Chroma vector store instance.
    """
    embeddings = get_embedding_model(model_name=embedding_model_name)

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    else:
        # In-memory ephemeral vector store for instant session use
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name
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
