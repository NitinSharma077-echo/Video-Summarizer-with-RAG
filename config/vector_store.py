import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "vector_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "transcript_collection")

os.makedirs(CHROMA_PATH, exist_ok=True)


def get_embedding():
    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_api_key and openai_api_key != "YOUR_API_KEY":
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
        )
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder="./model",
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API key missing and local embeddings not available: {e}")


def get_chroma_client():
    api_key = (os.getenv("CHROMA_API_KEY") or os.getenv("Vector_DB") or "").strip()
    tenant = (os.getenv("CHROMA_TENANT") or "52fb452f-1e38-40f1-a701-d99c7dd5eecc").strip()
    database = (os.getenv("CHROMA_DATABASE") or "RAG").strip()

    if api_key and api_key != "YOUR_API_KEY":
        try:
            return chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database,
            )
        except Exception as e:
            print(f"Warning: Failed to connect to Chroma CloudClient ({e}). Falling back to local persistent store.")
            return None
    return None


def delete_vector_store():
    """
    Deletes the current transcript collection in Chroma DB (Cloud or Local)
    so conversation history and document vectors are completely wiped out when conversation ends.
    """
    client = get_chroma_client()
    try:
        if client:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        else:
            embeddings = get_embedding()
            vs = Chroma(
                persist_directory=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
            vs.delete_collection()
    except Exception as e:
        print(f"Note: Vector store deletion notice: {e}")


def build_vector_store(transcript: str):
    # Always delete existing conversation collection first so old embeddings are deleted
    delete_vector_store()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]
    embeddings = get_embedding()
    client = get_chroma_client()

    if client:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            client=client,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
        )
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embedding()
    client = get_chroma_client()
    if client:
        return Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k},
    )


def search_vector_store(query, k=4):
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k)
    return retriever.invoke(query)


def hybrid_search(query, k=4, weight=0.5):
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k)
    return retriever.invoke(query)
