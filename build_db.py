from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import load_and_chunk_pdf # Importing the function we wrote earlier!

def create_vector_db():
    file_path = "sample_doc.pdf"
    
    # 1. Get the chunks from our first script
    print("Step 1: Ingesting and chunking document...")
    try:
        chunks = load_and_chunk_pdf(file_path)
    except Exception as e:
        print(f"Error loading PDF. Did you name it correctly? Error: {e}")
        return

    # 2. Initialize the Embedding Model
    # This converts our text into dense numerical vectors
    print("\nStep 2: Loading Embedding Model (This might take a minute the first time to download)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Build and save the Chroma Database
    print("\nStep 3: Building Vector Database...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db" # This saves the DB locally in a folder
    )
    
    print("\n✅ SUCCESS! Vector database built and saved to the './chroma_db' folder.")

if __name__ == "__main__":
    create_vector_db()