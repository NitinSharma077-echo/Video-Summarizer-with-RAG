import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_OLLAMA_MODELS = (
    "qwen2.5",
    "qwen2.5:1.5b",
    "qwen2.5:0.5b",
    "gemma4:e4b",
    "gemma3:1b",
    "qwen2.5:7b",
    "gemma3:4b",
    "gemma:4b",
    "phi4-mini:latest",
)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _configured_ollama_models() -> list[str]:
    raw_models = os.getenv("OLLAMA_MODELS") or os.getenv("OLLAMA_MODEL")
    if not raw_models:
        return list(DEFAULT_OLLAMA_MODELS)
    return [model.strip() for model in raw_models.split(",") if model.strip()]


def _installed_ollama_models() -> set[str]:
    url = f"{_ollama_base_url()}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return set()

    return {model.get("name", "") for model in payload.get("models", [])}


def _ollama_disabled() -> bool:
    return os.getenv("DISABLE_OLLAMA", "false").strip().lower() in {"1", "true", "yes"}


def _select_ollama_model() -> str | None:
    if _ollama_disabled():
        return None
    installed_models = _installed_ollama_models()
    if not installed_models:
        return None

    for model in _configured_ollama_models():
        if model in installed_models:
            return model
        matching_variants = sorted(
            installed_model
            for installed_model in installed_models
            if installed_model.startswith(f"{model}:")
        )
        if matching_variants:
            return matching_variants[0]

    completion_models = sorted(
        model
        for model in installed_models
        if not model.startswith("nomic-embed") and "embed" not in model
    )
    return completion_models[0] if completion_models else None


def get_llm():
    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_api_key and openai_api_key != "YOUR_API_KEY":
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.2,
        )

    ollama_model = _select_ollama_model()
    if ollama_model:
        return ChatOllama(
            model=ollama_model,
            base_url=_ollama_base_url(),
            temperature=0.2,
        )

    mistral_api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if mistral_api_key:
        mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        return ChatMistralAI(
            model=mistral_model,
            api_key=mistral_api_key,
            temperature=0.2,
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )
