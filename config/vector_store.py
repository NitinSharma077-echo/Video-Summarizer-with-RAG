import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_PATH="vector_store"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
COLLECTION_NAME="transcript_collection"

vector_path="vector_store"
os.makedirs(vector_path,exist_ok=True)  
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder="./model",
    )
    
def build_vector_store(transcript:str):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks=splitter.split_text(transcript)
    docs=[
        Document(page_content=chunk,metadata={'chunk_index':i})
        for i,chunk in enumerate(chunks)
    ]
    embeddings=get_embedding()

    # Reset any previously stored collection first so chunks from an
    # earlier video/audio file don't bleed into this session's RAG answers.
    try:
        existing=Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        existing.delete_collection()
    except Exception:
        pass

    vector_store=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )
    return vector_store

def load_vector_store()->Chroma:
    embeddings=get_embedding()
    return Chroma(persist_directory=CHROMA_PATH,embedding_function=embeddings,collection_name=COLLECTION_NAME)

def get_retriever(vector_store:Chroma,k:int=4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k":k}
    )

def search_vector_store(query,k=4):
    vector_store=load_vector_store()
    retriever=get_retriever(vector_store,k)
    return retriever.get_relevant_documents(query)

def hybrid_search(query,k=4,weight=0.5):
    vector_store=load_vector_store()
    retriever=get_retriever(vector_store,k)
    return retriever.get_relevant_documents(query)
    
