import os
import shutil
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Loads the environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in .env file.")

# Disables huggingface warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    This fucntion loads all PDF documents from a specified directory.
    Args:
        directory_path (str): Path to the directory containing PDF files.
    Returns:
        all_docs (List[Document]): List of Document objects containing the content of each PDF.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    all_docs: List[Document] = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata.setdefault("page", 0)
            all_docs.extend(docs)
    return all_docs


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    This function splits documents into smaller chunks for better processing.
    Args:
        documents (List[Document]): List of Document objects to be chunked.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between chunks.
    Returns:
        chunks (List[Document]): List of Document objects after chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        keep_separator=True
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def clear_vector_store(index_path: str) -> None:
    """
    This function clears the existing vector store at the specified path if needed.
    Args:
        index_path (str): Path to the vector store directory.
    Returns:
        None
    """
    abs_path = os.path.abspath(index_path)
    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)
        print(f"[INFO] Cleared existing vector store at '{abs_path}'")
    else:
        print(f"[INFO] No vector store found at '{abs_path}' to clear.")


def create_or_load_vector_store(chunks: List[Document], index_path: str) -> FAISS:
    """
    This function creates a new vector store if embeddings are not done or load an existing one from the specified path if embeddings are done.
    Args:
        chunks (List[Document]): List of Document objects to be indexed.
        index_path (str): Path to the vector store directory.
    Returns:
        vector_db (FAISS): The FAISS vector store containing the indexed documents.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    abs_path = os.path.abspath(index_path)
    if os.path.exists(abs_path):
        print(f"[INFO] Loading existing vector store from '{abs_path}'...")
        return FAISS.load_local(abs_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print(f"[INFO] Creating new vector store at '{abs_path}'...")
        vector_db = FAISS.from_documents(chunks, embedding_model)
        vector_db.save_local(abs_path)
        return vector_db


def build_qa_chain(vector_db: FAISS) -> RetrievalQA:
    """
    This function builds a question-answering chain using the provided vector store.
    Args:
        vector_db (FAISS): The FAISS vector store containing the indexed documents.
    Returns:
        qa_chain (RetrievalQA): The question-answering chain configured with the vector store.
    """
    prompt_template = """
You are an expert policy analyst. Answer the user's question using only the provided context.
If you don't know the answer, honestly say you don't know.
Include page references and source documents in your response.

Context:
{context}

Question: {question}

Answer:
"""
    custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
        max_tokens=1024
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}),
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )


def format_source_path(path: str) -> str:
    """
    Format the source path to display only the filename.
    Args:
        path (str): The full path to the source file.
    Returns:
        str: The formatted filename.
    """
    return os.path.basename(path)


def display_result(response: dict) -> None:
    """
    Display the result of the question-answering chain.
    Args:
        response (dict): The response from the question-answering chain.
    Returns:
        None
    """
    print("\nAnswer:")
    print(response.get("result", "").strip())

    print("\nSource References:")
    seen = set()
    for i, doc in enumerate(response.get("source_documents", [])):
        key = f"{doc.metadata.get('source_file')}-{doc.metadata.get('page')}"
        if key in seen:
            continue
        seen.add(key)

        print(f"\nSource {i+1}:")
        print(f"- Document: {format_source_path(doc.metadata.get('source_file', 'unknown'))}")
        print(f"- Page: {doc.metadata.get('page', 'N/A') + 1}")
        print(f"- Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
        content = doc.page_content
        if len(content) > 300:
            content = content[:50] + " [...] " + content[-50:]
        print(f"- Snippet:\n{content}\n")
