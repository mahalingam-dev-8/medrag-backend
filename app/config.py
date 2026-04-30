from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8000"])

    # Database — required, no default
    database_url: str = Field(...)

    # Groq
    groq_api_key: str = Field(...)
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")

    # Embedding model — safe defaults (not sensitive)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # Chunking — safe defaults
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # Retrieval — safe defaults
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.3)
    retrieval_candidate_k: int = Field(default=20)
    hybrid_search_enabled: bool = Field(default=True)

    # Reranker — safe default (public model name)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM generation — safe defaults
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_max_tokens: int = Field(default=1024)
    llm_temperature: float = Field(default=0.1)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
