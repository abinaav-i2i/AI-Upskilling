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

if __name__ == "__main__":
    if not DATA_DIR or not INDEX_PATH:
        raise ValueError("DATA_DIR and INDEX_PATH must be set in .env file")
    
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(project_root, "data")
    # index_path = "vector_store"
    
    #step 1: Clear existing vector store if needed
    clear_vector_store(INDEX_PATH)

    # Step 2: Load and chunk PDF documents
    all_docs = load_pdfs_from_directory(DATA_DIR)
    chunked_documents = chunk_documents(all_docs)

    # Step 3: Create or load the vector store
    vector_db = create_or_load_vector_store(chunked_documents, INDEX_PATH)

    # Step 4: Build the QA chain
    qa_chain = build_qa_chain(vector_db)

    # Step 5: Start the interactive Q&A session
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        display_result(result)
