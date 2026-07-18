from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from config.llm import get_llm
import os

MAX_SUMMARY_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", "6"))
MAX_TITLE_CHARS = int(os.getenv("TITLE_MAX_CHARS", "4000"))

def split_transcript(transcript):
    splitter=RecursiveCharacterTextSplitter(
        separators=[".", "\n"],
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def summarize(transcript):
    llm=get_llm()
    prompt_template=ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpful assistant that summarizes text"),
            ("user","""Give me a bullet point summary of the following transcript chunk:
            {transcript}
            """),
        ]
    )
    map_chain=prompt_template|llm|StrOutputParser()
    chunks=split_transcript(transcript)
    if len(chunks) > MAX_SUMMARY_CHUNKS:
        step = max(1, len(chunks) // MAX_SUMMARY_CHUNKS)
        chunks = chunks[::step][:MAX_SUMMARY_CHUNKS]
    chunk_summaries=[map_chain.invoke({"transcript":chunk}) for chunk in chunks]
    combine="\n\n".join(chunk_summaries)
    combined_prompt=ChatPromptTemplate.from_messages(
        [
            ("system","""You are an expert summarizer. Your task is to produce a comprehensive yet concise bullet point summary based on the provided chunk summaries.
            Instructions:
            - Synthesize the information logically
            - Preserve all key details
            - Do not add any information not present in the chunks
            - Use clear, professional language
            - Organize with bullet points
            """),
            ("user","""Here are the combined chunk summaries. Please create a final, well-organized bullet point summary:
            {combine}
            """)
        ]
    )

    combined_chain=combined_prompt|llm|StrOutputParser()
    return combined_chain.invoke({"combine": combine})

def generate_title(transcript:str):
    llm=get_llm()
    title_chain=(
        RunnablePassthrough()|ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert title generator. Your task is to produce a catchy and informative title based on the provided transcript.
                Instructions:
                - Come up with a creative title for the transcript.
                - Keep it concise and engaging.
                - Do not add any information not present in the transcript.
                - Use clear, professional language.
                """),
                ("user","""Here is the transcript. Please create a catchy and informative title:
                {transcript}
                """),
            ]
        )|llm|StrOutputParser()
    )
    return title_chain.invoke({"transcript": transcript[:MAX_TITLE_CHARS]})

