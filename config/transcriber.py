import os
from openai import OpenAI
import whisper

whisper_model = os.getenv("WHISPER_MODEL", "base")
_model = None


def load_local_model():
    global _model
    if _model is None:
        _model = whisper.load_model(whisper_model)
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> dict:
    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_api_key and openai_api_key != "YOUR_API_KEY":
        try:
            client = OpenAI(api_key=openai_api_key)
            with open(chunk_path, "rb") as audio_file:
                if translate:
                    response = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                else:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                return {"text": response.text}
        except Exception as e:
            print(f"OpenAI Whisper API notice: {e}. Falling back to local Whisper model.")

    model = load_local_model()
    result = model.transcribe(chunk_path, task="transcribe" if not translate else "translate")
    return result


def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_text = ""
    for chunk_path in chunks:
        result = transcribe_chunk(chunk_path, translate)
        full_text += result.get("text", "") + "\n"
    return full_text
