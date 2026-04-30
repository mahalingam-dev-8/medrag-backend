#!/usr/bin/env python3
"""RAG evaluation script (Phase 4).

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --dataset evaluation/datasets/medical_qa.json
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import httpx

from evaluation.metrics import evaluate_batch, summary_stats
from app.utils.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("evaluate_rag")

BASE_URL = "http://localhost:8000"


async def run_evaluation(dataset_path: Path) -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)

    questions = [item["question"] for item in dataset]
    expected = [" ".join(item.get("expected_keywords", [])) for item in dataset]
    actuals: list[str] = []
    chunks_per_query: list[list[str]] = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        for item in dataset:
            try:
                response = await client.post(
                    "/api/v1/chat/",
                    json={"question": item["question"]},
                )
                response.raise_for_status()
                data = response.json()
                actuals.append(data["answer"])
                chunks_per_query.append([s["content"] for s in data.get("sources", [])])
            except Exception as exc:
                logger.error("eval_request_failed", question=item["question"], error=str(exc))
                actuals.append("")
                chunks_per_query.append([])

    results = evaluate_batch(questions, expected, actuals, chunks_per_query)
    stats = summary_stats(results)

    print("\n=== RAG Evaluation Results ===")
    for r in results:
        print(f"\nQ: {r.question}")
        print(f"  Faithfulness: {r.faithfulness_score:.2f}")

    print(f"\nSummary: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/medical_qa.json"),
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.dataset))


if __name__ == "__main__":
    main()
