class MedRAGError(Exception):
    """Base exception for all med-rag-assistant errors."""


class DocumentNotFoundError(MedRAGError):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document not found: {document_id}")
        self.document_id = document_id


class ChunkNotFoundError(MedRAGError):
    def __init__(self, chunk_id: str) -> None:
        super().__init__(f"Chunk not found: {chunk_id}")
        self.chunk_id = chunk_id


class IngestionError(MedRAGError):
    """Raised when document ingestion fails."""


class EmbeddingError(MedRAGError):
    """Raised when embedding generation fails."""


class RetrievalError(MedRAGError):
    """Raised when vector retrieval fails."""


class GenerationError(MedRAGError):
    """Raised when LLM generation fails."""


class RerankingError(MedRAGError):
    """Raised when cross-encoder reranking fails."""


class SessionNotFoundError(MedRAGError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class UnsupportedFileTypeError(IngestionError):
    def __init__(self, file_type: str) -> None:
        super().__init__(f"Unsupported file type: {file_type}")
        self.file_type = file_type
