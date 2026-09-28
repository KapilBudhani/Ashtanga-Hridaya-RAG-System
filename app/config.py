from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Local development
load_dotenv(ROOT / ".env")


def get_secret(name: str, default=None):
    """
    Read a setting from Streamlit secrets when deployed,
    otherwise fall back to environment variables.
    """

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]

    except Exception:
        pass

    return os.getenv(name, default)


OCR_DIR = ROOT / "data" / "ocr_pages"
CHUNKS_DIR = ROOT / "data" / "chunks"
VECTOR_DIR = ROOT / "vectorstore"

COLLECTION_NAME = "ashtanga_sangraha_vol1"

EMBEDDING_MODEL = get_secret(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = get_secret(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

GROQ_API_KEY = get_secret(
    "GROQ_API_KEY"
)

OCR_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)