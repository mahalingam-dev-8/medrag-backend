#!/usr/bin/env python3
"""CLI batch ingestion script.

Usage:
    python scripts/ingest_docs.py data/raw/
    python scripts/ingest_docs.py path/to/doc.pdf
    python scripts/ingest_docs.py data/raw/ --chunk-size 256 --chunk-overlap 32
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from app.db.database import AsyncSessionLocal
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.pipeline import IngestionPipeline
from app.utils.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("ingest_docs")

SUPPORTED = {".pdf", ".txt", ".md"}


async def run(
    target: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    paths: list[Path] = []
    if target.is_dir():
        paths = [p for p in sorted(target.rglob("*")) if p.suffix.lower() in SUPPORTED]
    elif target.is_file():
        paths = [target]
    else:
        logger.error("path_not_found", path=str(target))
        sys.exit(1)

    if not paths:
        logger.warning("no_files_found", path=str(target))
        return

    logger.info("ingestion_start", files=len(paths))
    success = failed = skipped = 0

    async with AsyncSessionLocal() as session:
        pipeline = IngestionPipeline(
            doc_repo=DocumentRepository(session),
            chunk_repo=ChunkRepository(session),
        )
        for path in paths:
            try:
                result = await pipeline.run_from_path(
                    path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                await session.commit()
                if result.skipped:
                    skipped += 1
                    logger.info("skipped", file=path.name)
                else:
                    success += 1
                    logger.info(
                        "ingested",
                        file=path.name,
                        chunks=result.total_chunks,
                        document_id=result.document_id,
                    )
            except Exception as exc:
                await session.rollback()
                failed += 1
                logger.error("failed", file=str(path), error=str(exc))

    logger.info(
        "ingestion_summary",
        success=success,
        skipped=skipped,
        failed=failed,
        total=len(paths),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest documents into med-rag-assistant")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(run(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
