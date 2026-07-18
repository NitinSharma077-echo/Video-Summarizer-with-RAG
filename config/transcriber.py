import whisper
import os

whisper_model=os.getenv("WHISPER_MODEL", "base")
_model=None
def load_model():
    global _model
    if _model is None:
        _model=whisper.load_model(whisper_model)
    return _model

def transcribe_chunk(chunk_path,translate=False):
    model=load_model()
    result=model.transcribe(chunk_path,task="transcribe" if not translate else "translate")
    return result

def transcribe_all(chunks:list,translate:bool=False):
    full_text=""
    for chunk_path in chunks:
        result=transcribe_chunk(chunk_path,translate)
        full_text+=result["text"]+"\n"
    return full_text


