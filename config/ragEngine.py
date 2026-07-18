from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm import get_llm
from config.vector_store import load_vector_store, get_retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    vector_store=load_vector_store()
    retriever=get_retriever(vector_store,k=4)
    llm=get_llm()
    rag_template=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions about the transcript."),
            ("user", """
            Question: {question}
            Context: {context}
            """),
        ]
    )
    rag_chain=({"context": retriever|format_docs, "question": lambda x: x}|rag_template|llm|StrOutputParser())
    return rag_chain

def load_rag_chain():
    vector_store=load_vector_store()
    retriever=get_retriever(vector_store,k=4)
    llm=get_llm()
    rag_template=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions about the transcript."),
            ("user", """
            Question: {question}
            Context: {context}
            """),
        ]
    )
    rag_chain=({"context": retriever|format_docs, "question": lambda x: x}|rag_template|llm|StrOutputParser())
    return rag_chain

def rag_query(query:str):
    rag_chain=load_rag_chain()
    return rag_chain.invoke(query)    

def rag_answer(query:str):
    return rag_query(query)

