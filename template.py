"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))



# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired OpenAI LLM (gpt-4o-mini) by default,
    with fallback to word-overlap heuristics when offline or without API key.
    """

    def __init__(self, use_llm: bool = True, model: str = "gpt-4o-mini") -> None:
        self.use_llm = use_llm
        self.model = model


    def _call_openai_eval(self, prompt: str) -> float | None:
        if not self.use_llm:
            return None
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert RAG Evaluation Judge. "
                            "Return ONLY a JSON object with keys 'score' (float between 0.0 and 1.0) "
                            "and 'reasoning' (string)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            val = float(data.get("score", 0.0))
            return max(0.0, min(1.0, round(val, 3)))
        except Exception:
            return None


    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Measure how grounded the answer is in the context using gpt-4o-mini."""
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        prompt = (
            f"Metric: Faithfulness / Groundedness\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            f"Evaluate if every statement in the answer is grounded in the context. "
            f"Score 1.0 for fully grounded or valid safety refusal; lower for ungrounded claims."
        )
        llm_score = self._call_openai_eval(prompt)
        if llm_score is not None:
            return llm_score

        context_tokens = _tokenize(context)
        overlap = answer_tokens & context_tokens
        score = len(overlap) / len(answer_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """Measure how relevant the answer is to the question using gpt-4o-mini."""
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        prompt = (
            f"Metric: Answer Relevance\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Evaluate how directly and completely the answer responds to the question. "
            f"Score 1.0 for a complete direct answer or valid safety refusal."
        )
        llm_score = self._call_openai_eval(prompt)
        if llm_score is not None:
            return llm_score

        answer_tokens = _tokenize(answer)
        overlap = answer_tokens & question_tokens
        score = len(overlap) / len(question_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """Measure how well the answer covers the expected answer using gpt-4o-mini."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        prompt = (
            f"Metric: Answer Completeness\n"
            f"Expected Reference Answer: {expected}\n"
            f"Actual Answer: {answer}\n"
            f"Evaluate how completely the actual answer covers the key points of the reference answer."
        )
        llm_score = self._call_openai_eval(prompt)
        if llm_score is not None:
            return llm_score

        answer_tokens = _tokenize(answer)
        overlap = answer_tokens & expected_tokens
        score = len(overlap) / len(expected_tokens)
        return max(0.0, min(1.0, score))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by retrieved chunks."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_text = "\n".join(contexts)
        prompt = (
            f"Metric: Context Recall\n"
            f"Expected Reference Answer: {expected}\n"
            f"Retrieved Chunks:\n{union_text}\n"
            f"Evaluate how much of the reference answer facts are present in the retrieved chunks."
        )
        llm_score = self._call_openai_eval(prompt)
        if llm_score is not None:
            return llm_score

        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens.update(_tokenize(chunk))
        overlap = expected_tokens & union_tokens
        score = len(overlap) / len(expected_tokens)
        return max(0.0, min(1.0, score))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K)."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        union_text = "\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(contexts)])
        prompt = (
            f"Metric: Context Precision (Rank-Aware AP@K)\n"
            f"Expected Reference Answer: {expected}\n"
            f"Retrieved Chunks:\n{union_text}\n"
            f"Evaluate if the most relevant chunks are placed at top positions (Chunk 1-2)."
        )
        llm_score = self._call_openai_eval(prompt)
        if llm_score is not None:
            return llm_score

        relevant_flags = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            cov = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            relevant_flags.append(cov >= relevance_threshold)

        total_relevant = sum(1 for is_rel in relevant_flags if is_rel)
        if total_relevant == 0:
            return 0.0

        sum_p = 0.0
        rel_so_far = 0
        for k, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                rel_so_far += 1
                precision_at_k = rel_so_far / k
                sum_p += precision_at_k

        ap = sum_p / total_relevant
        return max(0.0, min(1.0, ap))


    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = (faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5)

        failure_type: str | None = None
        if faithfulness < 0.3:
            failure_type = "hallucination"
        elif relevance < 0.3:
            failure_type = "irrelevant"
        elif completeness < 0.3:
            failure_type = "incomplete"
        elif not passed:
            failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None

        if contexts is not None:
            reranked = rerank_by_overlap(contexts, question)
            context_recall = self.evaluate_context_recall(reranked, expected)
            context_precision = self.evaluate_context_precision(reranked, expected)

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else [],
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.
    """
    query_tokens = _tokenize(query)
    return sorted(contexts, key=lambda c: len(_tokenize(c) & query_tokens), reverse=True)


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.
        """
        rubric_str = json.dumps(rubric, indent=2)
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rubric: {rubric_str}\n"
            "Evaluate the answer based on the rubric and return JSON format."
        )

        raw_response = self.judge_llm_fn(prompt)
        scores: dict[str, float] = {}

        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    scores = {k: float(v) for k, v in parsed["scores"].items()}
                else:
                    scores = {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
        except Exception:
            pass

        if not scores:
            scores = {k: 0.5 for k in rubric}

        return {
            "scores": scores,
            "reasoning": raw_response,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria
        """
        all_scores = []
        for item in scores_batch:
            scores_dict = item.get("scores", {})
            for val in scores_dict.values():
                try:
                    all_scores.append(float(val))
                except (ValueError, TypeError):
                    pass

        avg_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.5

        leniency_bias = avg_score > 0.8
        severity_bias = avg_score < 0.3
        positional_bias = False
        if len(scores_batch) > 1:
            first_scores = [v for v in scores_batch[0].get("scores", {}).values()]
            rest_scores = [v for b in scores_batch[1:] for v in b.get("scores", {}).values()]
            if first_scores and rest_scores:
                avg_first = sum(first_scores) / len(first_scores)
                avg_rest = sum(rest_scores) / len(rest_scores)
                if avg_first - avg_rest > 0.2:
                    positional_bias = True

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.
        """
        results: list[EvalResult] = []
        total_count = len(qa_pairs)
        for idx, pair in enumerate(qa_pairs, start=1):
            item_id = pair.metadata.get("id", f"QA{idx:02d}") if pair.metadata else f"QA{idx:02d}"
            diff = pair.metadata.get("difficulty", "normal") if pair.metadata else "normal"
            print(f"[{idx}/{total_count}] Evaluating Test {item_id:<4} ({diff:<11}): {pair.question[:60]}...", flush=True)
            actual_answer = agent_fn(pair.question)
            contexts_to_pass = pair.retrieved_contexts if pair.retrieved_contexts else None
            eval_res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=contexts_to_pass,
            )
            eval_res.qa_pair = pair
            results.append(eval_res)
        return results


    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / total
        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevance = sum(r.relevance for r in results) / total
        avg_completeness = sum(r.completeness for r in results) / total

        recalls = [r.context_recall for r in results if r.context_recall is not None]
        precisions = [r.context_precision for r in results if r.context_precision is not None]

        avg_context_recall = sum(recalls) / len(recalls) if recalls else None
        avg_context_precision = sum(precisions) / len(precisions) if precisions else None

        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_context_recall": avg_context_recall,
            "avg_context_precision": avg_context_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        """Compare new evaluation results against a baseline.
        """
        n_count = len(new_results)
        b_count = len(baseline_results)

        new_f = sum(r.faithfulness for r in new_results) / n_count if n_count > 0 else 0.0
        new_r = sum(r.relevance for r in new_results) / n_count if n_count > 0 else 0.0
        new_c = sum(r.completeness for r in new_results) / n_count if n_count > 0 else 0.0

        base_f = sum(r.faithfulness for r in baseline_results) / b_count if b_count > 0 else 0.0
        base_r = sum(r.relevance for r in baseline_results) / b_count if b_count > 0 else 0.0
        base_c = sum(r.completeness for r in baseline_results) / b_count if b_count > 0 else 0.0

        regressions: list[str] = []
        if base_f - new_f > 0.05:
            regressions.append("faithfulness")
        if base_r - new_r > 0.05:
            regressions.append("relevance")
        if base_c - new_c > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.
        """
        return [
            r for r in results
            if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.
        """
        counts: dict[str, int] = {}
        for f in failures:
            if f.failure_type:
                counts[f.failure_type] = counts.get(f.failure_type, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.
        """
        f = failure.faithfulness
        r = failure.relevance
        c = failure.completeness

        min_score = min(f, r, c)
        if min_score == f and f < 0.5:
            return "Context is missing or irrelevant — improve retrieval"
        elif min_score == r and r < 0.5:
            return "Answer does not address the question — improve prompt clarity"
        elif min_score == c and c < 0.5:
            return "Answer is missing key information — increase context window or improve generation"
        else:
            return "Multiple issues detected — review full pipeline"

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.
        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for i, f in enumerate(failures, start=1):
            fid = f"F{i:03d}"
            ftype = f.failure_type if f.failure_type else "Unknown"
            cause = self.find_root_cause(f)
            fix = suggestions[i - 1] if i - 1 < len(suggestions) else (suggestions[0] if suggestions else "Review pipeline")
            lines.append(f"| {fid} | {ftype} | {cause} | {fix} | Open |")
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.
        """
        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        if categories.get("hallucination", 0) > 0:
            suggestions.append("Implement hallucination checker to filter unsupported claims")
        if categories.get("incomplete", 0) > 0:
            suggestions.append("Increase chunk size in RAG pipeline to reduce context fragmentation")
        if categories.get("irrelevant", 0) > 0 or categories.get("off_topic", 0) > 0:
            suggestions.append("Add few-shot examples showing complete answers to improve completeness")

        default_suggestions = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims",
        ]

        for ds in default_suggestions:
            if len(suggestions) >= 3:
                break
            if ds not in suggestions:
                suggestions.append(ds)

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)

