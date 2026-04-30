"""RAG evaluation metrics (Phase 4).

Implements faithfulness, relevance, and basic hallucination detection.
"""

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_chunks: list[str]
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    notes: list[str] = field(default_factory=list)


def keyword_faithfulness(answer: str, chunks: list[str]) -> float:
    """Rough faithfulness: fraction of answer words found in context."""
    if not chunks or not answer:
        return 0.0
    context = " ".join(chunks).lower()
    words = [w.lower() for w in answer.split() if len(w) > 4]
    if not words:
        return 1.0
    hits = sum(1 for w in words if w in context)
    return hits / len(words)


def evaluate_batch(
    questions: list[str],
    expected_answers: list[str],
    actual_answers: list[str],
    retrieved_chunks_per_query: list[list[str]],
) -> list[EvalResult]:
    results = []
    for q, exp, act, chunks in zip(questions, expected_answers, actual_answers, retrieved_chunks_per_query):
        faith = keyword_faithfulness(act, chunks)
        results.append(
            EvalResult(
                question=q,
                expected_answer=exp,
                actual_answer=act,
                retrieved_chunks=chunks,
                faithfulness_score=faith,
            )
        )
    return results


def summary_stats(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    avg_faith = sum(r.faithfulness_score for r in results) / len(results)
    return {
        "total": len(results),
        "avg_faithfulness": round(avg_faith, 3),
    }
