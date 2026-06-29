from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def query_rag(user_query):
    # 1. Initialize the same model we used to build the DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to our local database
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    
    # 3. Perform a semantic search
    print(f"\nQuerying for: '{user_query}'...")
    results = db.similarity_search(user_query, k=2) # k=2 returns top 2 relevant chunks
    
    print("\n--- AI Search Results ---")
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:\n{res.page_content}")

if __name__ == "__main__":
    question = input("What do you want to ask your PDF? ")
    query_rag(question)