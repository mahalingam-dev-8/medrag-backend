SYSTEM_PROMPT = """\
You are a knowledgeable medical assistant. Answer questions using the provided context \
from medical documents when available.

Guidelines:
- Prefer information from the provided context over general knowledge.
- When answering from context, cite the source document and page number.
- Use clear, professional medical language appropriate for healthcare providers.
- Never provide direct medical advice or diagnosis to patients.
"""

SYSTEM_PROMPT_NO_CONTEXT = """\
You are a knowledgeable medical assistant. Answer the question directly from your \
general medical knowledge. Do not mention documents, context, or say phrases like \
"based on the context" or "no information was provided". Just answer the question directly.

Guidelines:
- Use clear, professional medical language appropriate for healthcare providers.
- Never provide direct medical advice or diagnosis to patients.
"""

RAG_TEMPLATE = """\
Context from medical documents:
{context}

---
Question: {question}

Answer based on the context above:"""

NO_CONTEXT_TEMPLATE = """\
Question: {question}

Answer:"""


def build_rag_prompt(context_chunks: list[dict], question: str) -> str:
    if not context_chunks:
        return NO_CONTEXT_TEMPLATE.format(question=question)

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page_number")
        page_info = f", page {page}" if page else ""
        context_parts.append(
            f"[{i}] Source: {source}{page_info}\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)
    return RAG_TEMPLATE.format(context=context, question=question)


def get_system_prompt(has_context: bool) -> str:
    return SYSTEM_PROMPT if has_context else SYSTEM_PROMPT_NO_CONTEXT
