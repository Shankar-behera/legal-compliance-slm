"""
Runs the same benchmark set against three checkpoints — base model,
post-DAPT, post-SFT — and produces the comparison table for the
portfolio README: perplexity, exact match, F1, hallucination rate,
and per-example latency.

This extends src/eval/eval_hallucination.py rather than replacing it:
that module owns the hallucination/citation-grounding logic, this one
owns the cross-checkpoint comparison and the extra metrics (perplexity,
EM, F1, latency) needed for a full benchmark table.

Benchmark format (data/eval/compliance_benchmark.jsonl):
    {
        "prompt": "...",
        "reference_answer": "...",
        "reference_clauses": ["GDPR Art. 5", ...]
    }

Usage:
    python -m src.eval.benchmark \
        --base_model_path Qwen/Qwen2.5-1.5B-Instruct \
        --dapt_model_path /content/drive/.../checkpoints/dapt/final \
        --sft_model_path /content/drive/.../checkpoints/sft/final_adapter \
        --benchmark_path data/eval/compliance_benchmark.jsonl \
        --output_path docs/benchmark_results.md
"""
import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.eval_hallucination import extract_citations, load_benchmark


@dataclass
class BenchmarkRow:
    stage: str
    model_path: str
    perplexity: float
    exact_match: float
    f1: float
    hallucination_rate: float
    avg_latency_ms: float


def compute_perplexity(model, tokenizer, examples: List[dict]) -> float:
    """Average per-token perplexity over reference answers, teacher-forced."""
    losses = []
    for ex in examples:
        text = f"{ex['prompt']}\n{ex['reference_answer']}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device
        )
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
    avg_loss = sum(losses) / max(len(losses), 1)
    return float(torch.exp(torch.tensor(avg_loss)))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_with_latency(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()
    return response, latency_ms


def run_benchmark_for_checkpoint(
    stage: str,
    model_path: str,
    benchmark_path: str,
    adapter_path: Optional[str] = None,
) -> BenchmarkRow:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    examples = load_benchmark(benchmark_path)

    em_scores, f1_scores, latencies = [], [], []
    hallucinated = 0

    for ex in examples:
        response, latency_ms = generate_with_latency(model, tokenizer, ex["prompt"])
        em_scores.append(compute_exact_match(response, ex["reference_answer"]))
        f1_scores.append(compute_f1(response, ex["reference_answer"]))
        latencies.append(latency_ms)

        cited = set(extract_citations(response))
        reference_clauses = set(ex.get("reference_clauses", []))
        if cited - reference_clauses:
            hallucinated += 1

    n = max(len(examples), 1)
    perplexity = compute_perplexity(model, tokenizer, examples)

    return BenchmarkRow(
        stage=stage,
        model_path=model_path,
        perplexity=round(perplexity, 3),
        exact_match=round(sum(em_scores) / n, 4),
        f1=round(sum(f1_scores) / n, 4),
        hallucination_rate=round(hallucinated / n, 4),
        avg_latency_ms=round(sum(latencies) / n, 1),
    )


def rows_to_markdown_table(rows: List[BenchmarkRow]) -> str:
    header = (
        "| Stage | Perplexity | Exact Match | F1 | Hallucination Rate | Avg Latency (ms) |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r.stage} | {r.perplexity} | {r.exact_match} | {r.f1} | "
        f"{r.hallucination_rate} | {r.avg_latency_ms} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--dapt_model_path", required=True)
    parser.add_argument("--sft_model_path", required=True, help="LoRA adapter path")
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_path", default="docs/benchmark_results.md")
    parser.add_argument(
        "--output_json_path",
        default="docs/benchmark_results.json",
        help="Also write raw numbers as JSON for programmatic use",
    )
    args = parser.parse_args()

    rows = [
        run_benchmark_for_checkpoint("Base model", args.base_model_path, args.benchmark_path),
        run_benchmark_for_checkpoint("After DAPT", args.dapt_model_path, args.benchmark_path),
        run_benchmark_for_checkpoint(
            "After SFT",
            args.dapt_model_path,
            args.benchmark_path,
            adapter_path=args.sft_model_path,
        ),
    ]

    table = rows_to_markdown_table(rows)
    print(table)

    with open(args.output_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(table)

    with open(args.output_json_path, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)


if __name__ == "__main__":
    main()
