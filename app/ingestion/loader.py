import io
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.utils.exceptions import UnsupportedFileTypeError
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class LoadedPage:
    content: str
    page_number: int
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    filename: str
    source: str
    doc_type: str
    pages: list[LoadedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.content for p in self.pages)


def load_file(path: Path | str) -> LoadedDocument:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(suffix)

    logger.info("loading_file", path=str(path), type=suffix)

    if suffix == ".pdf":
        return _load_pdf(path)
    return _load_text(path)


def load_bytes(data: bytes, filename: str) -> LoadedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(suffix)

    if suffix == ".pdf":
        return _load_pdf_bytes(data, filename)
    return _load_text_bytes(data, filename)


def _load_pdf(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(content=text.strip(), page_number=i + 1))

    return LoadedDocument(
        filename=path.name,
        source=str(path),
        doc_type="pdf",
        pages=pages,
        metadata={"total_pages": len(reader.pages)},
    )


def _load_pdf_bytes(data: bytes, filename: str) -> LoadedDocument:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(content=text.strip(), page_number=i + 1))

    return LoadedDocument(
        filename=filename,
        source=filename,
        doc_type="pdf",
        pages=pages,
        metadata={"total_pages": len(reader.pages)},
    )


def _load_text(path: Path) -> LoadedDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    doc_type = "md" if path.suffix == ".md" else "txt"
    return LoadedDocument(
        filename=path.name,
        source=str(path),
        doc_type=doc_type,
        pages=[LoadedPage(content=content.strip(), page_number=1)],
    )


def _load_text_bytes(data: bytes, filename: str) -> LoadedDocument:
    content = data.decode("utf-8", errors="replace")
    suffix = Path(filename).suffix.lower()
    doc_type = "md" if suffix == ".md" else "txt"
    return LoadedDocument(
        filename=filename,
        source=filename,
        doc_type=doc_type,
        pages=[LoadedPage(content=content.strip(), page_number=1)],
    )
