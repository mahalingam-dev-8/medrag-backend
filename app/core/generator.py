from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.prompts import build_rag_prompt, get_system_prompt
from app.utils.exceptions import GenerationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Generator:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
        return self._client

    def _build_messages(
        self,
        user_prompt: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        messages = [{"role": "system", "content": get_system_prompt(bool(context_chunks))}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def generate(
        self,
        question: str,
        context_chunks: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
    ) -> str:
        _model = model or settings.llm_model
        _max_tokens = max_tokens or settings.llm_max_tokens
        _temperature = temperature if temperature is not None else settings.llm_temperature

        user_prompt = build_rag_prompt(context_chunks, question)
        logger.info("generating_response", model=_model, context_count=len(context_chunks))

        try:
            response = await self.client.chat.completions.create(
                model=_model,
                messages=self._build_messages(user_prompt, context_chunks, history),
                max_tokens=_max_tokens,
                temperature=_temperature,
            )
            answer = response.choices[0].message.content or ""
            logger.info(
                "generation_complete",
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
            return answer
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc

    async def stream(
        self,
        question: str,
        context_chunks: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        _model = model or settings.llm_model
        _max_tokens = max_tokens or settings.llm_max_tokens
        _temperature = temperature if temperature is not None else settings.llm_temperature

        user_prompt = build_rag_prompt(context_chunks, question)
        logger.info("streaming_response", model=_model, context_count=len(context_chunks))

        try:
            response = await self.client.chat.completions.create(
                model=_model,
                messages=self._build_messages(user_prompt, context_chunks, history),
                max_tokens=_max_tokens,
                temperature=_temperature,
                stream=True,
            )
            async for chunk in response:
                token = chunk.choices[0].delta.content
                if token is not None:
                    yield token
        except Exception as exc:
            raise GenerationError(f"LLM streaming failed: {exc}") from exc
