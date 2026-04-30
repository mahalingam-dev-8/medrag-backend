from dataclasses import dataclass, field

import tiktoken

from app.ingestion.loader import LoadedDocument, LoadedPage
from app.utils.logger import get_logger

logger = get_logger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    page_number: int | None = None
    section_title: str | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Split a loaded document into overlapping token-bounded chunks."""
    all_chunks: list[TextChunk] = []
    chunk_index = 0

    for page in document.pages:
        page_chunks = _split_text(
            text=page.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page.page_number,
            start_index=chunk_index,
        )
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    logger.info(
        "document_chunked",
        filename=document.filename,
        pages=len(document.pages),
        chunks=len(all_chunks),
    )
    return all_chunks


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page_number: int | None,
    start_index: int,
) -> list[TextChunk]:
    """Recursive character-based splitter respecting token limits."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _recursive_split(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        page_number=page_number,
        start_index=start_index,
    )


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    page_number: int | None,
    start_index: int,
) -> list[TextChunk]:
    tokens = count_tokens(text)
    if tokens <= chunk_size:
        return [
            TextChunk(
                content=text.strip(),
                chunk_index=start_index,
                page_number=page_number,
                token_count=tokens,
            )
        ] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if separator and separator in text:
        splits = text.split(separator)
    else:
        # Fall through to next separator
        if remaining_separators:
            return _recursive_split(
                text, chunk_size, chunk_overlap, remaining_separators, page_number, start_index
            )
        # Last resort: hard split by tokens
        return _hard_split(text, chunk_size, chunk_overlap, page_number, start_index)

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    idx = start_index

    for part in splits:
        part_tokens = count_tokens(part + separator)
        if current_tokens + part_tokens > chunk_size and current_parts:
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        chunk_index=idx,
                        page_number=page_number,
                        token_count=count_tokens(chunk_text),
                    )
                )
                idx += 1
            # Overlap: keep last N tokens worth of parts
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current_parts):
                pt = count_tokens(p + separator)
                if overlap_tokens + pt > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += pt
            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(part)
        current_tokens += part_tokens

    if current_parts:
        chunk_text = separator.join(current_parts).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_index=idx,
                    page_number=page_number,
                    token_count=count_tokens(chunk_text),
                )
            )

    return chunks


def _hard_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page_number: int | None,
    start_index: int,
) -> list[TextChunk]:
    tokens = _tokenizer.encode(text)
    chunks: list[TextChunk] = []
    idx = start_index
    pos = 0

    while pos < len(tokens):
        end = min(pos + chunk_size, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text = _tokenizer.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_index=idx,
                    page_number=page_number,
                    token_count=len(chunk_tokens),
                )
            )
            idx += 1
        pos += chunk_size - chunk_overlap

    return chunks
