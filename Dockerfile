FROM python:3.11-slim

# ffmpeg is required by pydub and by the direct ffmpeg subprocess calls used
# for chunking uploaded audio/video. Render's native Python runtime does not
# guarantee ffmpeg is present, so this Docker image installs it explicitly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=base

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image so the first request doesn't pay a
# multi-minute cold-start download (and still works if outbound downloads
# are ever restricted at request time).
RUN python -c "import whisper; whisper.load_model('base')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model')"

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
