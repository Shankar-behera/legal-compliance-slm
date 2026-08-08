"""
Computes and reports dataset statistics for both training phases:
  - DAPT corpus: document/block count, total tokens, avg block length
  - SFT pairs: instruction pair count, avg prompt/response token lengths

Run this after chunk_text.py and build_sft_dataset.py, before training,
so the numbers in your README/model card are measured, not estimated.

Usage:
    python -m data.scripts.dataset_stats \
        --dapt_path data/processed/dapt_chunks.jsonl \
        --sft_path data/processed/sft_pairs.jsonl \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --output_path docs/dataset_stats.md
"""
import argparse
import json
from dataclasses import asdict, dataclass
from typing import Optional

from transformers import AutoTokenizer


@dataclass
class DaptStats:
    n_blocks: int
    total_tokens: int
    avg_block_tokens: float


@dataclass
class SftStats:
    n_pairs: int
    avg_prompt_tokens: float
    avg_response_tokens: float
    total_tokens: int


def compute_dapt_stats(path: str, tokenizer) -> DaptStats:
    n_blocks = 0
    total_tokens = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_tokens = len(tokenizer.encode(record["text"], add_special_tokens=False))
            total_tokens += n_tokens
            n_blocks += 1

    avg_block_tokens = total_tokens / n_blocks if n_blocks else 0.0
    return DaptStats(n_blocks=n_blocks, total_tokens=total_tokens, avg_block_tokens=round(avg_block_tokens, 1))


def compute_sft_stats(path: str, tokenizer) -> SftStats:
    n_pairs = 0
    prompt_tokens_total = 0
    response_tokens_total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt_tokens_total += len(tokenizer.encode(record["prompt"], add_special_tokens=False))
            response_tokens_total += len(
                tokenizer.encode(record["response"], add_special_tokens=False)
            )
            n_pairs += 1

    avg_prompt = prompt_tokens_total / n_pairs if n_pairs else 0.0
    avg_response = response_tokens_total / n_pairs if n_pairs else 0.0
    return SftStats(
        n_pairs=n_pairs,
        avg_prompt_tokens=round(avg_prompt, 1),
        avg_response_tokens=round(avg_response, 1),
        total_tokens=prompt_tokens_total + response_tokens_total,
    )


def format_report(dapt_stats: Optional[DaptStats], sft_stats: Optional[SftStats]) -> str:
    lines = ["# Dataset Statistics\n"]

    if dapt_stats:
        lines.append("## DAPT Corpus\n")
        lines.append(f"- Blocks: {dapt_stats.n_blocks:,}")
        lines.append(f"- Total tokens: {dapt_stats.total_tokens:,}")
        lines.append(f"- Avg tokens/block: {dapt_stats.avg_block_tokens}\n")

    if sft_stats:
        lines.append("## SFT Instruction Pairs\n")
        lines.append(f"- Pairs: {sft_stats.n_pairs:,}")
        lines.append(f"- Avg prompt tokens: {sft_stats.avg_prompt_tokens}")
        lines.append(f"- Avg response tokens: {sft_stats.avg_response_tokens}")
        lines.append(f"- Total tokens: {sft_stats.total_tokens:,}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapt_path", default=None)
    parser.add_argument("--sft_path", default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", default="docs/dataset_stats.md")
    parser.add_argument("--output_json_path", default="docs/dataset_stats.json")
    args = parser.parse_args()

    if not args.dapt_path and not args.sft_path:
        raise ValueError("Provide at least one of --dapt_path or --sft_path")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dapt_stats = compute_dapt_stats(args.dapt_path, tokenizer) if args.dapt_path else None
    sft_stats = compute_sft_stats(args.sft_path, tokenizer) if args.sft_path else None

    report = format_report(dapt_stats, sft_stats)
    print(report)

    with open(args.output_path, "w") as f:
        f.write(report)

    with open(args.output_json_path, "w") as f:
        json.dump(
            {
                "dapt": asdict(dapt_stats) if dapt_stats else None,
                "sft": asdict(sft_stats) if sft_stats else None,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
