"""
Evaluates the fine-tuned model against a fixed benchmark set of compliance
QA pairs, scoring both keyword-grounded accuracy and a hallucination proxy
(fabricated clause/statute citations not present in the benchmark's
reference answer). This produces the baseline-vs-fine-tuned comparison
number referenced in the portfolio write-up — run it against the base
model AND the fine-tuned model to get a real delta, not an assumed one.

Benchmark format (data/eval/compliance_benchmark.jsonl):
    {"prompt": "...", "reference_clauses": ["GDPR Art. 5", ...], "reference_answer": "..."}

Usage:
    python -m src.eval.eval_hallucination \
        --model_path <path-or-hf-id> \
        --benchmark_path data/eval/compliance_benchmark.jsonl \
        --output_path docs/benchmark_results.json
"""
import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalResult:
    model_path: str
    n_examples: int
    clause_grounding_rate: float  # fraction of generations citing a reference clause
    fabricated_citation_rate: float  # fraction citing a clause NOT in reference set
    avg_response_length: float


CITATION_PATTERN = re.compile(
    r"\b([A-Z]{2,10}\s?(?:Art\.?|Article|§)\s?[\d\.\(\)a-zA-Z]+)", re.IGNORECASE
)


def extract_citations(text: str) -> List[str]:
    return [m.strip() for m in CITATION_PATTERN.findall(text)]


def load_benchmark(path: str) -> List[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


def evaluate(model_path: str, benchmark_path: str) -> EvalResult:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()

    examples = load_benchmark(benchmark_path)
    grounded = 0
    fabricated = 0
    total_len = 0

    for ex in examples:
        response = generate(model, tokenizer, ex["prompt"])
        total_len += len(response.split())

        cited = set(extract_citations(response))
        reference = set(ex.get("reference_clauses", []))

        if cited & reference:
            grounded += 1
        if cited - reference:
            fabricated += 1

    n = max(len(examples), 1)
    return EvalResult(
        model_path=model_path,
        n_examples=len(examples),
        clause_grounding_rate=grounded / n,
        fabricated_citation_rate=fabricated / n,
        avg_response_length=total_len / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_path", default="docs/benchmark_results.json")
    args = parser.parse_args()

    result = evaluate(args.model_path, args.benchmark_path)
    print(json.dumps(asdict(result), indent=2))

    with open(args.output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)


if __name__ == "__main__":
    main()
