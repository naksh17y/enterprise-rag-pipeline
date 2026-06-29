from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path):
    print(f"Loading document: {file_path}...")
    
    # 1. Load the PDF document
    loader = PyPDFLoader(file_path)
    document = loader.load()
    
    print(f"Document loaded successfully. Pages: {len(document)}")
    
    # 2. Configure the Text Splitter
    # We use overlapping chunks so we don't cut a sentence in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # The max number of characters in a chunk
        chunk_overlap=50,     # How many characters overlap between chunks
        length_function=len,
        is_separator_regex=False,
    )
    
    # 3. Split the document into chunks
    chunks = text_splitter.split_documents(document)
    print(f"Document split into {len(chunks)} chunks.")
    
    # Display the first chunk as a test
    print("\n--- First Chunk Preview ---")
    print(chunks[0].page_content)
    print("---------------------------\n")
    
    return chunks

# Test the function (Replace 'sample_doc.pdf' with a real PDF file)
if __name__ == "__main__":
    file_path = "sample_doc.pdf" 
    try:
        document_chunks = load_and_chunk_pdf(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please add a PDF to the directory.")