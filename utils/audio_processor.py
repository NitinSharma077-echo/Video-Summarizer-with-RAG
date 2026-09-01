import yt_dlp
from pydub import AudioSegment
from typing import Optional
import os
import json
import re
import subprocess
import urllib.request
from urllib.parse import urlparse
download_audio='downloads'
os.makedirs(download_audio,exist_ok=True)


def configure_ffmpeg():
    local_app_data = os.getenv("LOCALAPPDATA")
    candidate_paths = [
        os.getenv("FFMPEG_BINARY"),
        os.path.join(local_app_data, "Microsoft", "WinGet", "Links", "ffmpeg.exe") if local_app_data else None,
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            AudioSegment.converter = path
            AudioSegment.ffmpeg = path
            return


configure_ffmpeg()


def get_ffmpeg_binary() -> str:
    if AudioSegment.converter and os.path.exists(AudioSegment.converter):
        return AudioSegment.converter
    return os.getenv("FFMPEG_BINARY", "ffmpeg")


class QuietLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def download_youtube_audio(url: str) -> Optional[str]:
    output_path=os.path.join(download_audio, '%(id)s.%(ext)s')
    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True, "logger": QuietLogger()}) as ydl:
        info = ydl.extract_info(url, download=False)
        cached_mp3 = os.path.join(download_audio, f"{info['id']}.mp3")
        if os.path.exists(cached_mp3):
            return cached_mp3

    ydl_opts={
        "format": "bestaudio/best",
        "outtmpl":output_path,
        "postprocessors":[
            {
                "key":"FFmpegExtractAudio",
                "preferredcodec":"mp3",
                "preferredquality":"192"
            }
        ],
        "quiet":True,
        "noprogress": True,
        "no_warnings":True,
        "logger": QuietLogger(),
        "progress_hooks": [],
        "windowsfilenames": True,
        "ignoreerrors":False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url,download=True)
        downloaded_path=ydl.prepare_filename(info)
        filename=os.path.splitext(downloaded_path)[0]+".mp3"
        return filename


def _subtitle_candidates(info: dict):
    preferred_languages = ("en", "en-US", "en-GB")
    subtitle_groups = (info.get("subtitles") or {}, info.get("automatic_captions") or {})

    for subtitles in subtitle_groups:
        for language in preferred_languages:
            if language in subtitles:
                yield from subtitles[language]
        for tracks in subtitles.values():
            yield from tracks


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _json3_to_text(raw_text: str) -> str:
    payload = json.loads(raw_text)
    lines = []
    for event in payload.get("events", []):
        text = "".join(segment.get("utf8", "") for segment in event.get("segs", []))
        text = " ".join(text.split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def _vtt_to_text(raw_text: str) -> str:
    lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped == "WEBVTT"
            or "-->" in stripped
            or stripped.startswith(("Kind:", "Language:", "NOTE"))
            or re.fullmatch(r"\d+", stripped)
        ):
            continue
        cleaned = re.sub(r"<[^>]+>", "", stripped)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(cleaned)
    return "\n".join(dict.fromkeys(lines))


def get_youtube_transcript(url: str) -> Optional[str]:
    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True, "logger": QuietLogger()}) as ydl:
        info = ydl.extract_info(url, download=False)

    transcript_path = os.path.join(download_audio, f"{info['id']}_transcript.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as handle:
            cached_transcript = handle.read().strip()
        if cached_transcript:
            return cached_transcript

    candidates = list(_subtitle_candidates(info))
    candidates.sort(key=lambda item: 0 if item.get("ext") == "json3" else 1)
    for candidate in candidates:
        subtitle_url = candidate.get("url")
        if not subtitle_url:
            continue
        raw_text = _download_text(subtitle_url)
        if candidate.get("ext") == "json3":
            transcript = _json3_to_text(raw_text)
        else:
            transcript = _vtt_to_text(raw_text)
        transcript = transcript.strip()
        if transcript:
            with open(transcript_path, "w", encoding="utf-8") as handle:
                handle.write(transcript)
            return transcript

    return None


def convert_to_wave(input_path:str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video or audio file not found: {input_path}")
    output_path=os.path.splitext(input_path)[0]+"_converted.wav"
    audio=AudioSegment.from_file(input_path)
    audio=audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path,format="wav")
    return output_path

def chunk_audio(wav_path,chunk_minutes=5):
    audio=AudioSegment.from_file(wav_path)
    chunk_duration_ms=chunk_minutes*60*1000
    chunks=[]
    for i,start in enumerate(range(0,len(audio),chunk_duration_ms)):
        chunk=audio[start:start+chunk_duration_ms]
        chunk_path=f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path,format="wav")
        chunks.append(chunk_path)
    return chunks


def create_wav_chunks(input_path: str, chunk_minutes: int = 5):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video or audio file not found: {input_path}")

    chunk_pattern = os.path.splitext(input_path)[0] + "_chunk_%03d.wav"
    chunk_dir = os.path.dirname(input_path) or "."
    chunk_prefix = os.path.splitext(os.path.basename(input_path))[0] + "_chunk_"
    for filename in os.listdir(chunk_dir):
        if filename.startswith(chunk_prefix) and filename.endswith(".wav"):
            os.remove(os.path.join(chunk_dir, filename))

    command = [
        get_ffmpeg_binary(),
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "segment",
        "-segment_time", str(chunk_minutes * 60),
        "-reset_timestamps", "1",
        chunk_pattern,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed to process the file."
        raise RuntimeError(detail)

    chunks = [
        os.path.join(chunk_dir, filename)
        for filename in sorted(os.listdir(chunk_dir))
        if filename.startswith(chunk_prefix) and filename.endswith(".wav")
    ]
    if not chunks:
        raise RuntimeError("No audio chunks were created from the video.")
    return chunks
    
def cleanup_chunks(chunks):
    """Remove the temporary wav chunks written for transcription."""
    for chunk_path in chunks or []:
        try:
            os.remove(chunk_path)
        except OSError:
            pass


def process_start(source: str):
    parsed_source = urlparse(source)
    if parsed_source.scheme in {"http", "https"}:
        audio_path = download_youtube_audio(source)
    else:
        audio_path = source
    return audio_path, create_wav_chunks(audio_path)
    
