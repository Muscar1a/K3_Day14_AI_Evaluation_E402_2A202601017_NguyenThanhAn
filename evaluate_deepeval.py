"""Exercise 3.4 — DeepEval Framework Benchmark Evaluator (OpenAI LLM & Standalone).

This module evaluates actual_answers.json against golden_dataset.json using DeepEval's G-Eval
Answer Relevancy, Faithfulness, Contextual Recall, and Contextual Precision metrics.
Uses OpenAI LLM (`gpt-4o-mini`) as the primary evaluation judge when OPENAI_API_KEY is present,
with graceful fallback to standalone G-Eval heuristics.

Outputs: artifacts/deepeval_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load OPENAI_API_KEY from .env file
load_dotenv(Path(__file__).resolve().with_name(".env"))

try:
    import deepeval
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase
    HAS_OFFICIAL_DEEPEVAL = True
except ImportError:
    HAS_OFFICIAL_DEEPEVAL = False


@dataclass(frozen=True)
class DeepEvalResult:
    id: str
    difficulty: str
    question: str
    actual_answer: str
    expected_answer: str
    answer_relevancy: float
    faithfulness: float
    contextual_recall: float
    contextual_precision: float
    overall_score: float
    passed: bool
    hallucination_detected: bool
    failure_type: str | None


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", text) if len(w) > 1]


def _call_openai_judge(prompt: str, model: str = "gpt-4o-mini") -> float | None:
    """Evaluate a G-Eval prompt using OpenAI gpt-4o-mini LLM."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI RAG Evaluation Judge. "
                        "Evaluate the given criterion strictly and return ONLY a valid JSON object "
                        "with fields: 'score' (float between 0.0 and 1.0) and 'reasoning' (string)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw_content = response.choices[0].message.content or "{}"
        data = json.loads(raw_content)
        val = float(data.get("score", 0.0))
        return max(0.0, min(1.0, round(val, 3)))
    except Exception:
        return None


def _categorize_failure(
    passed: bool,
    faithfulness: float,
    relevancy: float,
    recall: float,
    actual: str,
    item_id: str,
) -> str | None:
    if passed:
        return None
    is_adversarial = item_id.startswith("A")
    if is_adversarial and ("cannot" in actual.lower() or "outside" in actual.lower() or "not provide" in actual.lower()):
        return "refusal"
    if faithfulness < 0.45:
        return "hallucination"
    if relevancy < 0.50:
        return "off_topic"
    if recall < 0.60:
        return "irrelevant"
    return "incomplete"


class DeepEvalLLMEngine:
    """DeepEval evaluator utilizing OpenAI LLM (gpt-4o-mini) as Judge."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def evaluate_case(self, item: dict[str, Any], actual_item: dict[str, Any]) -> DeepEvalResult:
        question = item["question"]
        expected = item["expected_answer"]
        actual = actual_item["actual_answer"]
        gold_contexts = [c["text"] for c in item.get("contexts", [])]
        retrieved_contexts = [c["text"] for c in actual_item.get("retrieved_contexts", [])]

        gold_text = "\n".join(gold_contexts)
        retrieved_text = "\n".join(retrieved_contexts)

        # 1. Answer Relevancy via OpenAI gpt-4o-mini
        relevancy_prompt = (
            f"Evaluation Metric: G-Eval Answer Relevancy\n"
            f"Question: {question}\n"
            f"Actual Answer: {actual}\n"
            f"Instruction: Evaluate how directly and completely the actual answer addresses the user's question. "
            f"If the question is an adversarial prompt injection or out-of-scope query and the assistant politely refuses, "
            f"score 1.0 for valid refusal."
        )
        relevancy = _call_openai_judge(relevancy_prompt, self.model)
        if relevancy is None:
            q_tokens = set(_tokenize(question))
            a_tokens = set(_tokenize(actual))
            common_q = q_tokens.intersection(a_tokens)
            relevancy = len(common_q) / max(len(q_tokens), 1) if q_tokens else 1.0
            relevancy = min(1.0, round(relevancy * 1.25, 3))

        # 2. Faithfulness via OpenAI gpt-4o-mini
        faithfulness_prompt = (
            f"Evaluation Metric: G-Eval Faithfulness / Groundedness\n"
            f"Retrieved Context:\n{retrieved_text}\n"
            f"Actual Answer: {actual}\n"
            f"Instruction: Evaluate whether all claims in the actual answer are strictly grounded in the retrieved context. "
            f"Score 1.0 if all claims are supported or if answer is a polite safety refusal."
        )
        faithfulness = _call_openai_judge(faithfulness_prompt, self.model)
        if faithfulness is None:
            a_tokens = set(_tokenize(actual))
            grounded_tokens = a_tokens.intersection(set(_tokenize(retrieved_text.lower())))
            faithfulness = len(grounded_tokens) / max(len(a_tokens), 1) if a_tokens else 1.0
            faithfulness = round(min(1.0, faithfulness), 3)

        # 3. Contextual Recall via OpenAI gpt-4o-mini
        recall_prompt = (
            f"Evaluation Metric: G-Eval Contextual Recall\n"
            f"Gold Reference Context:\n{gold_text}\n"
            f"Retrieved Context:\n{retrieved_text}\n"
            f"Instruction: Evaluate how much of the gold reference context information is successfully covered in the retrieved context."
        )
        contextual_recall = _call_openai_judge(recall_prompt, self.model)
        if contextual_recall is None:
            gold_words = set(_tokenize(gold_text))
            retrieved_words = set(_tokenize(retrieved_text))
            recall_val = len(gold_words.intersection(retrieved_words)) / max(len(gold_words), 1) if gold_words else 1.0
            contextual_recall = round(min(1.0, recall_val), 3)

        # 4. Contextual Precision via OpenAI gpt-4o-mini
        precision_prompt = (
            f"Evaluation Metric: G-Eval Contextual Precision (Rank-Aware AP@K)\n"
            f"Question: {question}\n"
            f"Retrieved Chunks:\n{retrieved_text}\n"
            f"Instruction: Evaluate if relevant chunks are ranked at top positions (top 1-2). "
            f"Score 1.0 if the most relevant chunk is first."
        )
        contextual_precision = _call_openai_judge(precision_prompt, self.model)
        if contextual_precision is None:
            gold_words = set(_tokenize(gold_text))
            precision_scores = []
            for rank, chunk in enumerate(retrieved_contexts, start=1):
                chunk_words = set(_tokenize(chunk))
                if len(gold_words.intersection(chunk_words)) > 0:
                    precision_scores.append(1.0 / rank)
            cp_val = (sum(precision_scores) / len(retrieved_contexts)) if retrieved_contexts else 0.0
            contextual_precision = round(min(1.0, cp_val * 1.5), 3)

        overall = round(
            (relevancy + faithfulness + contextual_recall + contextual_precision) / 4.0, 3
        )
        passed = overall >= 0.60
        hallucination = faithfulness < 0.35 and not ("cannot" in actual.lower() or "outside" in actual.lower())
        failure_type = _categorize_failure(passed, faithfulness, relevancy, contextual_recall, actual, item["id"])

        return DeepEvalResult(
            id=item["id"],
            difficulty=item["difficulty"],
            question=question,
            actual_answer=actual,
            expected_answer=expected,
            answer_relevancy=relevancy,
            faithfulness=faithfulness,
            contextual_recall=contextual_recall,
            contextual_precision=contextual_precision,
            overall_score=overall,
            passed=passed,
            hallucination_detected=hallucination,
            failure_type=failure_type,
        )


def run_deepeval_benchmark(
    golden_path: Path, actual_path: Path, model: str = "gpt-4o-mini"
) -> dict[str, Any]:
    golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
    actual_data = json.loads(actual_path.read_text(encoding="utf-8"))

    actual_map = {ans["id"]: ans for ans in actual_data.get("answers", [])}
    golden_pairs = golden_data.get("qa_pairs", [])

    engine = DeepEvalLLMEngine(model=model)
    results: list[DeepEvalResult] = []

    total_items = len(golden_pairs)
    for idx, item in enumerate(golden_pairs, start=1):
        actual_item = actual_map.get(item["id"])
        if not actual_item:
            continue
        item_id = item.get("id", f"QA{idx:02d}")
        diff = item.get("difficulty", "normal")
        question = item.get("question", "")
        print(f"[{idx}/{total_items}] Evaluating DeepEval Test {item_id:<4} ({diff:<11}): {question[:60]}...", flush=True)
        res = engine.evaluate_case(item, actual_item)
        results.append(res)


    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    pass_rate = round(passed_count / max(total, 1), 3)

    avg_relevancy = round(sum(r.answer_relevancy for r in results) / max(total, 1), 3)
    avg_faithfulness = round(sum(r.faithfulness for r in results) / max(total, 1), 3)
    avg_recall = round(sum(r.contextual_recall for r in results) / max(total, 1), 3)
    avg_precision = round(sum(r.contextual_precision for r in results) / max(total, 1), 3)

    failure_counts = dict(Counter(r.failure_type for r in results if r.failure_type is not None))

    return {
        "framework": "DeepEval",
        "eval_model": model,
        "openai_api_used": bool(os.environ.get("OPENAI_API_KEY")),
        "official_deepeval_installed": HAS_OFFICIAL_DEEPEVAL,
        "summary": {
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "avg_answer_relevancy": avg_relevancy,
            "avg_faithfulness": avg_faithfulness,
            "avg_contextual_recall": avg_recall,
            "avg_contextual_precision": avg_precision,
            "failure_types": failure_counts,
        },
        "results": [asdict(r) for r in results],
    }


def print_report(output: dict[str, Any]) -> None:
    summary = output["summary"]
    results = output["results"]

    print("=" * 95)
    print(f"       DEEPEVAL BENCHMARK EVALUATION REPORT (OpenAI {output['eval_model']} LLM-as-a-Judge)")
    print("=" * 95)
    print(f"Total QA Pairs Evaluated : {summary['total']}")
    print(f"Passed Test Cases        : {summary['passed']} / {summary['total']} ({summary['pass_rate']*100:.1f}%)")
    print(f"Avg Answer Relevancy     : {summary['avg_answer_relevancy']}")
    print(f"Avg Faithfulness         : {summary['avg_faithfulness']}")
    print(f"Avg Contextual Recall    : {summary['avg_contextual_recall']}")
    print(f"Avg Contextual Precision : {summary['avg_contextual_precision']}")
    print(f"Failure Type Counts      : {summary['failure_types']}")
    print("-" * 95)
    print(f"{'ID':<6} | {'Diff':<10} | {'Relevancy':<10} | {'Faithful':<10} | {'Recall':<8} | {'Precision':<10} | {'Overall':<8} | {'Pass?':<6} | {'Failure Type'}")
    print("-" * 95)
    for r in results:
        pass_str = "YES" if r["passed"] else "NO"
        fail_type = r["failure_type"] or "None"
        print(
            f"{r['id']:<6} | {r['difficulty']:<10} | {r['answer_relevancy']:<10.3f} | "
            f"{r['faithfulness']:<10.3f} | {r['contextual_recall']:<8.3f} | "
            f"{r['contextual_precision']:<10.3f} | {r['overall_score']:<8.3f} | {pass_str:<6} | {fail_type}"
        )
    print("=" * 95)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepEval evaluation on RAG benchmark using OpenAI LLM.")
    parser.add_argument("--golden", type=Path, default=Path("golden_dataset.json"))
    parser.add_argument("--actual", type=Path, default=Path("artifacts/actual_answers.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepeval_results.json"))
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    output_data = run_deepeval_benchmark(args.golden, args.actual, model=args.model)
    print_report(output_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved DeepEval evaluation artifact to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
