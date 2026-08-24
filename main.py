from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Iterable
from urllib.parse import urlparse
from config.transcriber import transcribe_all
from utils.audio_processor import get_youtube_transcript, process_start
from config.translator import summarize, generate_title
from config.extractor import extract_action_items, extract_key_moments, extract_key_points
from config.vector_store import build_vector_store, delete_vector_store
from config.ragEngine import rag_answer
from config.pdf_export import build_transcript_pdf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

# Single active-session state: this is a single-tenant demo app, so the most
# recently analyzed video/audio is what /chat, /download/transcript and
# /download/media operate on.
SESSION = {
    "title": "",
    "transcript": "",
    "summary": "",
    "action_items": "",
    "key_moments": "",
    "key_points": "",
    "media_path": None,
    "media_name": None,
}


def validate_extension(filename: str, allowed_extensions: Iterable[str] = ALLOWED_EXTENSIONS) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="Uploaded file must have a video or audio extension.")
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use one of: {allowed}")
    return suffix


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]", "_", name or "video").strip() or "video"
    return name[:120]


def analyze_source(source: str, media_path: str = None, media_name: str = None):
    parsed_source = urlparse(source)
    transcript = None
    if parsed_source.scheme in {"http", "https"}:
        transcript = get_youtube_transcript(source)

    if not transcript:
        audio_path, chunks = process_start(source)
        transcript = transcribe_all(chunks)
        # process_start() downloads YouTube audio to disk when captions are
        # unavailable, so that file becomes the downloadable media too.
        if media_path is None and parsed_source.scheme in {"http", "https"} and os.path.exists(audio_path):
            media_path = audio_path

    if not transcript or not transcript.strip():
        raise ValueError("Could not extract any speech from this source. Try a different file or URL.")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    key_moments = extract_key_moments(transcript)
    key_points = extract_key_points(transcript)

    build_vector_store(transcript)

    SESSION.update(
        title=title,
        transcript=transcript,
        summary=summary,
        action_items=action_items,
        key_moments=key_moments,
        key_points=key_points,
        media_path=media_path,
        media_name=media_name or (os.path.basename(media_path) if media_path else None),
    )

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_moments": key_moments,
        "key_points": key_points,
        "media_available": bool(media_path and os.path.exists(media_path)),
    }

@app.post("/process")
def process_video(request: VideoRequest):
    try:
        return analyze_source(request.url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    suffix = validate_extension(file.filename or "")
    upload_path = (UPLOAD_DIR / f"{uuid4().hex}{suffix}").resolve()
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if upload_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        return analyze_source(
            str(upload_path),
            media_path=str(upload_path),
            media_name=_safe_filename(file.filename or upload_path.name),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):
    if not SESSION["transcript"]:
        raise HTTPException(status_code=400, detail="Process a video or audio file first.")
    try:
        answer = rag_answer(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
@app.post("/clear")
def reset_session():
    delete_vector_store()
    SESSION.update(
        title="",
        transcript="",
        summary="",
        action_items="",
        key_moments="",
        key_points="",
        media_path=None,
        media_name=None,
    )
    return {"status": "success", "message": "Conversation and vector database cleared for new session."}



@app.get("/download/transcript")
def download_transcript():
    if not SESSION["transcript"]:
        raise HTTPException(status_code=404, detail="No transcript available yet. Process a video first.")

    buffer = build_transcript_pdf(
        title=SESSION["title"],
        transcript=SESSION["transcript"],
        summary=SESSION["summary"],
        key_points=SESSION["key_points"],
        action_items=SESSION["action_items"],
        key_moments=SESSION["key_moments"],
    )
    filename = f"{_safe_filename(SESSION['title'] or 'transcript')}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download/media")
def download_media():
    media_path = SESSION["media_path"]
    if not media_path or not os.path.exists(media_path):
        raise HTTPException(
            status_code=404,
            detail="No downloadable video/audio for this session (captions-only URL analysis doesn't store media).",
        )
    filename = SESSION["media_name"] or os.path.basename(media_path)
    return FileResponse(media_path, filename=filename, media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
