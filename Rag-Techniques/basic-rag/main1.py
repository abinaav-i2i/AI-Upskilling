import argparse
from rag_pipeline import (
    load_pdfs_from_directory,
    chunk_documents,
    create_or_load_vector_store,
    build_qa_chain,
    display_result,
    clear_vector_store
)
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_PATH = os.getenv("INDEX_PATH", "./vector_store")


def run_build():
    """
    Build the vector store from PDF documents and save it.
    This function clears the existing vector store, loads PDF documents from the specified directory,
    chunks them into manageable pieces, creates a new vector store, and saves it to the specified path.
    """
    clear_vector_store(INDEX_PATH)
    all_docs = load_pdfs_from_directory(DATA_DIR)
    chunked_documents = chunk_documents(all_docs)
    vector_db = create_or_load_vector_store(chunked_documents, INDEX_PATH)
    print("Vector store built and saved.")


def run_query():
    """
    Run the query interface to interact with the vector store.
    This function loads the vector store from the specified path and allows users to ask questions
    interactively, retrieving answers based on the indexed documents.
    """
    vector_db = create_or_load_vector_store([], INDEX_PATH)
    qa_chain = build_qa_chain(vector_db)
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        display_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument(
        "--mode", choices=["build", "query"], default="query",
        help="Mode to run the app: 'build' to process PDFs and save embeddings, 'query' to ask questions."
        )

    args = parser.parse_args()

    if args.mode == "build":
        run_build()
    elif args.mode == "query":
        run_query()
