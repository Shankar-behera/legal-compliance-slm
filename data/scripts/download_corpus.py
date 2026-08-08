"""
Pulls real regulatory/legal text from Hugging Face's `pile-of-law/pile-of-law`
dataset and writes it to `data/raw/dapt/*.txt`, where `chunk_text.py` picks
it up. This replaces manually sourcing DAPT text — you don't hand-collect
10-20M tokens, you stream them from an existing public legal corpus.

Streamed (`streaming=True`), so it never loads a full subset into memory —
required to stay inside Colab's ~12GB system RAM, same constraint
`chunk_text.py` is built around.

Default subsets are the ones most relevant to a *compliance* auditor
(rather than e.g. case law or contracts), out of pile-of-law's ~35
available configs:
    - cfr              Code of Federal Regulations
    - eurlex           EU legislation (GDPR-adjacent)
    - privacy_policies Real privacy policy text
    - tos              Real Terms of Service text (complements the
                        LexGLUE unfair_tos SFT source — DAPT sees the
                        vocabulary, SFT sees labeled violations of it)

Usage:
    python data/scripts/download_corpus.py \
        --subsets cfr eurlex privacy_policies tos \
        --max_docs_per_subset 1500 \
        --output_dir data/raw/dapt
"""
import argparse
import os
from typing import List

from datasets import load_dataset

AVAILABLE_SUBSETS = {
    "cfr",
    "eurlex",
    "privacy_policies",
    "tos",
    "uscode",
    "state_codes",
    "constitutions",
    "echr",
}

DEFAULT_SUBSETS = ["cfr", "eurlex", "privacy_policies", "tos"]

# Rough chars-per-token estimate for a stopping-criterion proxy only — the
# real, exact token count is computed later by chunk_text.py's tokenizer.
# This just avoids downloading far more raw text than the DAPT target
# (10-20M tokens) needs.
APPROX_CHARS_PER_TOKEN = 4


def download_subset(
    subset: str,
    output_dir: str,
    max_docs: int,
    max_chars: int,
    trust_remote_code: bool = True,
) -> int:
    if subset not in AVAILABLE_SUBSETS:
        raise ValueError(
            f"Unknown subset '{subset}'. Known compliance-relevant subsets: "
            f"{sorted(AVAILABLE_SUBSETS)}. See the pile-of-law dataset card "
            f"on Hugging Face for the full list if you want a different one."
        )

    dataset = load_dataset(
        "pile-of-law/pile-of-law",
        subset,
        split="train",
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    out_path = os.path.join(output_dir, f"{subset}.txt")
    n_docs = 0
    n_chars = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for example in dataset:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            out_f.write(text.strip() + "\n\n")
            n_docs += 1
            n_chars += len(text)

            if n_docs >= max_docs or n_chars >= max_chars:
                break

    return n_docs


def download_corpus(
    subsets: List[str],
    output_dir: str,
    max_docs_per_subset: int,
    max_tokens_per_subset: int,
    trust_remote_code: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    max_chars_per_subset = max_tokens_per_subset * APPROX_CHARS_PER_TOKEN

    for subset in subsets:
        print(f"Downloading '{subset}' (up to {max_docs_per_subset} docs / "
              f"~{max_tokens_per_subset:,} tokens)...")
        n_docs = download_subset(
            subset=subset,
            output_dir=output_dir,
            max_docs=max_docs_per_subset,
            max_chars=max_chars_per_subset,
            trust_remote_code=trust_remote_code,
        )
        print(f"  wrote {n_docs} documents to {output_dir}/{subset}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets", nargs="+", default=DEFAULT_SUBSETS,
        help=f"pile-of-law configs to pull. Known: {sorted(AVAILABLE_SUBSETS)}",
    )
    parser.add_argument("--output_dir", default="data/raw/dapt")
    parser.add_argument("--max_docs_per_subset", type=int, default=1500)
    parser.add_argument(
        "--max_tokens_per_subset", type=int, default=5_000_000,
        help="Approximate stopping point per subset — real token count is "
             "measured later by chunk_text.py, this is just to avoid "
             "over-downloading raw text.",
    )
    parser.add_argument(
        "--no_trust_remote_code", action="store_true",
        help="Disable trust_remote_code if your datasets version doesn't need it "
             "(newer parquet-based dataset versions on the Hub often don't).",
    )
    args = parser.parse_args()

    download_corpus(
        subsets=args.subsets,
        output_dir=args.output_dir,
        max_docs_per_subset=args.max_docs_per_subset,
        max_tokens_per_subset=args.max_tokens_per_subset,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
