"""RAG evaluation using RAGAS with OpenAI as judge.

Metrics:
  - faithfulness      : are claims in the answer supported by context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : were the retrieved chunks actually needed?

Cost: ~$0.005 for 15 samples using gpt-4o-mini.

Setup:
    pip install 'ragas>=0.2.0' langchain-openai
    Set OPENAI_API_KEY in .env
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    reference: str = ""


@dataclass
class EvalSummary:
    total: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "answer_correctness": round(self.answer_correctness, 3),
        }


def evaluate_ragas(samples: list[EvalSample]) -> EvalSummary:
    """Full RAGAS evaluation using OpenAI gpt-4o-mini as judge."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Run: pip install 'ragas>=0.2.0' langchain-openai"
        ) from exc

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    print("  Judge: OpenAI gpt-4o-mini (~$0.005 for 15 samples)")

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    )

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.reference or s.answer,
        )
        for s in samples
    ])

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=60, max_retries=3, max_wait=30),
    )

    df = results.to_pandas()
    return EvalSummary(
        total=len(df),
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_precision=float(df["context_precision"].mean()),
        context_recall=float(df["context_recall"].mean()),
        answer_correctness=float(df["answer_correctness"].mean()),
    )
