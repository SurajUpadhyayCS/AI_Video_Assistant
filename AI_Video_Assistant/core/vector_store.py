import os
from langchain.community import HUGGINGFACEEMBEDDINGS
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_core.document import Document
from langchain_chroma import Chroma

Chroma_dir = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding():
    return HUGGINGFACEEMBEDDINGS(
        model_name = EMBEDDING_MODEL
        model_kwargs = {"device" = "cpu"}
    )


def build_vectora_store():
    print("Building Vector Store")
    
    Splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    
    chunks = Splitter.split_text(transcript)
    
    docs = [
        Document(page_content=chunks,metadata= {'chunk_index: i'})
        for i,chunk in enumerate(chunks)
    ]
    
    EMBEDDINGS = get_embedding()
    
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding_function = EMBEDDINGS,
        persist_directory = Chroma_dir,
        collection_name = COLLECTION_NAME
    )
    
    return vector_store


def load_vector_store():
    EMBEDDINGS = get_embedding()
    
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding_function = EMBEDDINGS,
        persist_directory = Chroma_dir,
        collection_name = COLLECTION_NAME
    )
    
    return vector_store

def get_retriever(vector_store: Chroma , k : int = 4) :
    return vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":k}
    )
    