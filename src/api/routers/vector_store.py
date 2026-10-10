from fastapi import APIRouter, Depends, HTTPException, status

from app.clients import REPO_ROOT, logger
from app.vector_store import (
    PROCESSED_TRANSCRIPTS_DIR,
    app_params,
    chunk_overlap,
    chunk_size,
    upsert_documents,
    vs
)
from utils.transcript_utils import load_and_cleaned_transcripts

from api.schemas import ChunkCountResponse, TranscriptSyncResponce
from api.security import require_admin_key

router = APIRouter(
    prefix="/internal/vector-store",
    tags=["vector-store-admin"],
    dependencies=[Depends[require_admin_key]]
)

RAW_TRANSCRIPTS_DIR = REPO_ROOT

@router.get("/chunks/count", response_model= ChunkCountResponse)
def count_chunks() -> ChunkCountResponse:
    try:
        count = vs._collection.count()
    except Exception as e:
        logger.error(f"[vector-store] failed to count chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to count chunks in the vector store."
        ) from e
    return ChunkCountResponse(collection=app_params.collection_name, count=count)

@router.post("/transcripts/sync", response_model=TranscriptSyncResponce)
def sync_transcripts() -> TranscriptSyncResponce:
    try:
        load_and_cleaned_transcripts(RAW_TRANSCRIPTS_DIR, PROCESSED_TRANSCRIPTS_DIR)
        stats = upsert_documents(chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"[vector-store] failed to sync transcripts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to sync transcripts into the vector store"
        ) from e
    return TranscriptSyncResponce(**stats)

