import logging
import shutil
from logging import INFO
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = ROOT_DIR / "saved-embeddings"

logger = logging.getLogger(name="CI vector store builder")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)


def main() -> None:
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
        logger.info(f"Removed stale vector store at {VECTOR_STORE_DIR}")
    else:
        logger.info("No existing vector store to remove")

    from app.vector_store import (
        PROCESSED_TRANSCRIPTS_DIR,
        chunk_overlap,
        chunk_size,
        upsert_documents
    )
    from utils.transcript_utils import load_and_cleaned_transcripts

    RAW_TRANSCRIPTS_DIR = ROOT_DIR / "data" / "raw"

    load_and_cleaned_transcripts(RAW_TRANSCRIPTS_DIR, PROCESSED_TRANSCRIPTS_DIR)
    logger.info("Transcripts cleaned")

    stats = upsert_documents(chunk_size, chunk_overlap)
    logger.info(f"Vector store rebuilt: {stats}")


if __name__ == "__main__":
    main()