"""RAG chain module: Assembles strict retrieval-augmented generation pipeline using Groq Cloud."""

from typing import Dict, Any, Generator, Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

# Exact required fallback response
STRICT_FALLBACK_RESPONSE = "Information not found in the document."

# Strict System Prompt Template enforcing absolute adherence to context
STRICT_SYSTEM_PROMPT = """You are a strict, authoritative document question-answering assistant.

CRITICAL RULES:
1. Answer the user's question SOLELY and EXCLUSIVELY using the facts contained within the provided "DOCUMENT CONTEXT" below.
2. Absolutely DO NOT extrapolate, assume, interpolate, or use any pre-existing or outside world knowledge.
3. If the answer is not directly, unambiguously, and factually present in the provided "DOCUMENT CONTEXT", your entire answer MUST BE EXACTLY:
"{fallback}"
4. Do NOT apologize, do NOT provide polite preamble, and do NOT attempt partial guesses.
5. If the context contains the answer, be concise, factual, and cite the Page number when available.

DOCUMENT CONTEXT:
---------------------
{context}
---------------------"""


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks with clear sectioning and page metadata."""
    if not docs:
        return "No relevant context found."
    
    formatted_chunks = []
    for idx, doc in enumerate(docs):
        page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
        chunk_header = f"[Document Excerpt {idx + 1} | Page {page}]"
        formatted_chunks.append(f"{chunk_header}\n{doc.page_content.strip()}")
        
    return "\n\n".join(formatted_chunks)


def build_rag_chain(
    vector_store: Chroma,
    api_key: str,
    model_name: str = "llama-3.1-8b-instant",
    top_k: int = 4
):
    """
    Constructs a strict RAG retrieval chain using LangChain and ChatGroq.
    
    Args:
        vector_store: Initialized Chroma vector store.
        api_key: Groq API key.
        model_name: Model identifier (defaults to 'llama-3.1-8b-instant').
        top_k: Number of nearest chunks to retrieve.
        
    Returns:
        retriever, chain: Tuple containing the retriever and the runnable chain.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    # Temperature is strictly set to 0.0 for zero hallucination and determinism
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        groq_api_key=api_key,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", STRICT_SYSTEM_PROMPT.format(context="{context}", fallback=STRICT_FALLBACK_RESPONSE)),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": lambda x: format_docs(x["documents"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def execute_strict_rag_stream(
    query: str,
    retriever,
    chain
) -> Tuple[Generator[str, None, None], List[Document]]:
    """
    Executes similarity retrieval and streams the LLM response.
    
    Returns:
        Tuple: (stream_generator, retrieved_documents)
    """
    # 1. Retrieve the top-k chunks
    retrieved_docs = retriever.invoke(query)

    # 2. Invoke the chain as a token generator
    stream_gen = chain.stream({
        "question": query,
        "documents": retrieved_docs
    })

    return stream_gen, retrieved_docs
