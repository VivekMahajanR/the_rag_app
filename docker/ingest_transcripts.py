from app.clients import REPO_ROOT
from app.vector_store import (
    PROCESSED_TRANSCRIPTS_DIR,
    chunk_overlap,
    chunk_size,
    upsert_documents
)
from utils.transcript_utils import load_and_cleaned_transcripts

load_and_cleaned_transcripts(REPO_ROOT / "data" / "raw", PROCESSED_TRANSCRIPTS_DIR)
print("ingest:", upsert_documents(chunk_size, chunk_overlap))