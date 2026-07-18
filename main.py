from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Iterable
from urllib.parse import urlparse
from config.transcriber import transcribe_all
from utils.audio_processor import get_youtube_transcript, process_start
from config.translator import summarize, generate_title
from config.extractor import extract_action_items, extract_key_moments, extract_key_points
from config.vector_store import build_vector_store
from config.ragEngine import rag_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

class VideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    query: str

global_transcript = ""
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}


def validate_extension(filename: str, allowed_extensions: Iterable[str] = ALLOWED_EXTENSIONS) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="Uploaded file must have a video or audio extension.")
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use one of: {allowed}")
    return suffix


def analyze_source(source: str):
    parsed_source = urlparse(source)
    transcript = None
    if parsed_source.scheme in {"http", "https"}:
        transcript = get_youtube_transcript(source)

    if not transcript:
        chunks = process_start(source)
        transcript = transcribe_all(chunks)

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    key_moments = extract_key_moments(transcript)
    key_points = extract_key_points(transcript)

    build_vector_store(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_moments": key_moments,
        "key_points": key_points
    }

@app.post("/process")
def process_video(request: VideoRequest):
    global global_transcript
    try:
        result = analyze_source(request.url)
        global_transcript = result["transcript"]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    global global_transcript
    suffix = validate_extension(file.filename or "")
    upload_path = (UPLOAD_DIR / f"{uuid4().hex}{suffix}").resolve()
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if upload_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = analyze_source(str(upload_path))
        global_transcript = result["transcript"]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = rag_answer(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
