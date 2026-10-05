"""Document processing module: Extracts text from PDFs and chunks it recursively."""

import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(
    file_bytes_or_path, 
    file_name: str = "uploaded_doc.pdf",
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Loads a PDF from either a file path or in-memory byte buffer,
    extracts the text using PyPDFLoader, and splits it into chunks.
    
    Args:
        file_bytes_or_path: File path str or bytes/BytesIO-like object.
        file_name: Name of the uploaded file for metadata reference.
        chunk_size: Maximum characters per chunk (Fixed requirement: 1000).
        chunk_overlap: Number of overlapping characters (Fixed requirement: 200).
        
    Returns:
        List[Document]: List of chunked Document objects with updated metadata.
    """
    temp_file_path = None
    
    try:
        # If passed as bytes or a stream (e.g. from Streamlit UploadedFile)
        if hasattr(file_bytes_or_path, "read"):
            if hasattr(file_bytes_or_path, "seek"):
                file_bytes_or_path.seek(0)
            suffix = os.path.splitext(file_name)[-1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_bytes_or_path.read())
                temp_file_path = tmp_file.name
            target_path = temp_file_path
        elif isinstance(file_bytes_or_path, bytes):
            suffix = os.path.splitext(file_name)[-1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_bytes_or_path)
                temp_file_path = tmp_file.name
            target_path = temp_file_path
        else:
            target_path = str(file_bytes_or_path)

        # 1. Load document pages using PyPDFLoader
        loader = PyPDFLoader(target_path)
        raw_documents = loader.load()

        if not raw_documents:
            raise ValueError(f"No readable content could be extracted from {file_name}.")

        # Attach original document filename to metadata
        for doc in raw_documents:
            doc.metadata["source_file"] = file_name
            # Ensure page metadata is 1-indexed for human readability
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1

        # 2. Split documents recursively
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        chunks = text_splitter.split_documents(raw_documents)
        
        # Tag each chunk with its sequential chunk index
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    finally:
        # Clean up temporary file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
