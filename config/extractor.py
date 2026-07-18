from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from config.llm import get_llm
import os

MAX_EXTRACTION_CHARS = int(os.getenv("EXTRACTION_MAX_CHARS", "12000"))


def transcript_sample(transcript: str) -> str:
    if len(transcript) <= MAX_EXTRACTION_CHARS:
        return transcript
    half = MAX_EXTRACTION_CHARS // 2
    return (
        transcript[:half]
        + "\n\n[Middle of transcript omitted for faster dashboard extraction. Full transcript is still stored for RAG chat.]\n\n"
        + transcript[-half:]
    )

def build_chain(system_prompt):
    llm=get_llm()
    return (RunnablePassthrough()|RunnableLambda(lambda x:{"text":x})|ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("user","{text}"),
        ]
    )|llm|StrOutputParser()
    )

def extract_action_items(transcript:str):
    action_system_prompt="""You are an expert action item extractor. Your task is to identify and extract all action items from the following transcript.
    Instructions:
    - Identify all action items
    - Extract action items
    - Preserve all key details
    - Do not add any information not present in the transcript
    - Use clear, professional language
    - Organize with bullet points
    """
    chain=build_chain(action_system_prompt)
    return chain.invoke(transcript_sample(transcript))

def extract_key_moments(transcript:str):
    key_moments_system_prompt="""You are an expert key moments extractor. Your task is to identify and extract all key moments from the following transcript.
    Instructions:
    - Identify all key moments
    - Extract key moments
    - Preserve all key details
    - Do not add any information not present in the transcript
    - Use clear, professional language
    - Organize with bullet points
    """
    chain=build_chain(key_moments_system_prompt)
    return chain.invoke(transcript_sample(transcript))

def extract_key_points(transcript:str):
    key_points_system_prompt="""You are an expert key points extractor. Your task is to identify and extract all key points from the following transcript.
    Instructions:
    - Identify all key points
    - Extract key points
    - Preserve all key details
    - Do not add any information not present in the transcript
    - Use clear, professional language
    - Organize with bullet points
    """
    chain=build_chain(key_points_system_prompt)
    return chain.invoke(transcript_sample(transcript))

