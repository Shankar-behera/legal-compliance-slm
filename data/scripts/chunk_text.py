"""
Streaming text chunker for DAPT corpus preparation.

Reads raw legal text files (already extracted from PDFs/HTML), tokenizes
with the target model's tokenizer, and writes fixed-length token blocks to
a JSONL file — one JSON object per line: {"text": "<decoded block>"}.

Designed to run under Colab's ~12GB system RAM: never loads the full
corpus into memory. Reads line-by-line, tokenizes in a rolling buffer,
and flushes complete blocks to disk as soon as they're ready.

Usage:
    python data/scripts/chunk_text.py \
        --input_dir data/raw/dapt \
        --output_path data/processed/dapt_chunks.jsonl \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --block_size 512
"""
import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List

from transformers import AutoTokenizer


def iter_raw_files(input_dir: str) -> Iterator[str]:
    """Yield paths of .txt files under input_dir, sorted for reproducibility."""
    paths = sorted(Path(input_dir).rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found under {input_dir}. "
            f"Extract PDFs to plain text first (see build_sft_dataset.py "
            f"for the pdfplumber pattern)."
        )
    for p in paths:
        yield str(p)


def iter_lines(file_path: str) -> Iterator[str]:
    """Stream a text file line by line instead of loading it whole."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def chunk_corpus(
    input_dir: str,
    output_path: str,
    tokenizer_name: str,
    block_size: int = 512,
    min_block_size: int = 64,
) -> int:
    """
    Tokenize the corpus in a streaming fashion and write fixed-size token
    blocks (decoded back to text) to a JSONL file.

    Returns the number of blocks written.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    buffer: List[int] = []
    n_blocks = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_path in iter_raw_files(input_dir):
            for line in iter_lines(file_path):
                token_ids = tokenizer.encode(line, add_special_tokens=False)
                buffer.extend(token_ids)
                # keep a document boundary signal without blowing memory
                buffer.append(tokenizer.eos_token_id)

                while len(buffer) >= block_size:
                    block = buffer[:block_size]
                    buffer = buffer[block_size:]
                    text = tokenizer.decode(block, skip_special_tokens=False)
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    n_blocks += 1

        # flush a final partial block if it's substantial enough to be useful
        if len(buffer) >= min_block_size:
            text = tokenizer.decode(buffer, skip_special_tokens=False)
            out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_blocks += 1

    return n_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of raw .txt files")
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--min_block_size", type=int, default=64)
    args = parser.parse_args()

    n_blocks = chunk_corpus(
        input_dir=args.input_dir,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        block_size=args.block_size,
        min_block_size=args.min_block_size,
    )
    approx_tokens = n_blocks * args.block_size
    print(f"Wrote {n_blocks} blocks (~{approx_tokens:,} tokens) to {args.output_path}")


if __name__ == "__main__":
    main()
