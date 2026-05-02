"""Run RAGAS evaluation against the live API.

Usage:
    python evaluation/run_eval.py --api-url http://localhost:8000
    python evaluation/run_eval.py --api-url http://localhost:8000 --max-samples 5

Results saved to evaluation/results.json.
Requires OPENAI_API_KEY in .env (~$0.005 for 15 samples).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from evaluation.metrics import EvalSample, evaluate_ragas

DATASETS_DIR = Path(__file__).parent / "datasets"


async def ask(client: httpx.AsyncClient, api_url: str, question: str) -> tuple[str, list[str]]:
    resp = await client.post(
        f"{api_url}/api/v1/chat/",
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    contexts = [s.get("content", "") for s in data.get("sources", []) if s.get("content")]
    return data.get("answer", ""), contexts


async def run(api_url: str, dataset_path: Path, max_samples: int | None) -> None:
    with dataset_path.open() as f:
        dataset: list[dict] = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"Collecting answers from {api_url} ({len(dataset)} questions)...")
    samples: list[EvalSample] = []
    async with httpx.AsyncClient() as client:
        for item in dataset:
            question = item["question"]
            print(f"  asking: {question[:70]}...")
            try:
                answer, contexts = await ask(client, api_url, question)
                samples.append(EvalSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    reference=item.get("answer", ""),
                ))
            except Exception as exc:
                print(f"  SKIP: {exc}")

    if not samples:
        print("No samples collected — is the API running?")
        return

    print(f"\nScoring {len(samples)} samples with RAGAS...")
    summary = evaluate_ragas(samples)
    print(json.dumps(summary.as_dict(), indent=2))

    out = Path(__file__).parent / "results.json"
    with out.open("w") as f:
        json.dump(summary.as_dict(), f, indent=2)
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--dataset", default=str(DATASETS_DIR / "medical_qa.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.api_url, Path(args.dataset), args.max_samples))


if __name__ == "__main__":
    main()
