from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# load raw pdf(s)
DATA_PATH="data/"
def load_pdf_files(data) :
    try:
        if not os.path.exists(data):
            raise FileNotFoundError(f"Directory not found: {data}")
            
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No PDF files found in {data}")
            
        return documents
    except Exception as e:
        print(f"Error loading PDF files: {str(e)}")
        raise

documents=load_pdf_files(DATA_PATH)

# create chunks
def create_chunks(extracted_data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                       chunk_overlap=50 )
        text_chunks= text_splitter.split_documents(extracted_data)
        
        if not text_chunks:
            raise ValueError("No text chunks created from the documents")
            
        return text_chunks
    except Exception as e:
        print(f"Error creating chunks: {str(e)}")
        raise

text_chunks=create_chunks(documents)

# create vector embeddings
def get_embedding_model():
    try:
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise

embedding_model=get_embedding_model()


# store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"

# Main execution with error handling
try:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    db=FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("Vector store created successfully!")
    
except Exception as e:
    print(f"Error creating vector store: {str(e)}")
    raise