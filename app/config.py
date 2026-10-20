from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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

    # LLM: any OpenAI-compatible API. Defaults to Google Gemini's OpenAI endpoint; the older
    # GROQ_* variable names are still accepted so existing deployments keep working.
    llm_api_key: str = Field(
        ..., validation_alias=AliasChoices("LLM_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")
    )
    llm_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias=AliasChoices("LLM_BASE_URL", "GROQ_BASE_URL"),
    )

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
    llm_model: str = Field(default="gemini-2.5-flash")
    # Gemini's thinking tokens count against this limit, so leave room beyond the answer itself.
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.1)
    # Thinking depth for models that support it ("low" keeps answers fast); empty to omit,
    # e.g. for LLaMA models on Groq, which reject the parameter.
    llm_reasoning_effort: str = Field(default="low")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
