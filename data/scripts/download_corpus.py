"""
Download legal text from the Pile-of-Law repository for DAPT.

IMPORTANT:
The current Hugging Face `datasets` library no longer supports the
legacy pile-of-law.py loading script (`RuntimeError: Dataset scripts
are no longer supported`), and pile-of-law has no `refs/convert/parquet`
mirror to fall back on either (`RevisionNotFoundError` — the Hub never
generated one for this dataset). So this script bypasses `datasets`
entirely: it downloads the underlying .jsonl.xz files directly from the
Hugging Face dataset repository over HTTP and streams/decompresses them
line-by-line. The config->filename mapping below is copied from
pile-of-law's own loading script (`pile-of-law.py`'s `_DATA_URL` dict),
not guessed — verified against the dataset repo directly.

Output:
    data/raw/dapt/cfr.txt
    data/raw/dapt/eurlex.txt
    data/raw/dapt/tos.txt
    ...

Each JSONL record contains a `text` field.

Designed for Google Colab with limited RAM: streams and decompresses
one line at a time, never holds a full subset in memory, and never
downloads the .xz file to disk before decompressing it.

Usage:
    python data/scripts/download_corpus.py \
        --subsets cfr eurlex tos \
        --max_docs_per_subset 1500 \
        --output_dir data/raw/dapt
"""

import argparse
import json
import lzma
import os
import time
from typing import Dict, List, Tuple

import requests


# ---------------------------------------------------------------------
# Current Pile-of-Law repository
# ---------------------------------------------------------------------

BASE_URL = (
    "https://huggingface.co/datasets/"
    "pile-of-law/pile-of-law/resolve/main/data/"
)


# Config name -> filename, copied directly from pile-of-law.py's own
# _DATA_URL dict (single-file configs only — a few pile-of-law configs
# like atticus_contracts and courtlistener_opinions are sharded across
# multiple files and are intentionally not included here since none of
# them are relevant to a compliance DAPT corpus anyway).
#
# NOTE: privacy_policies is intentionally NOT included — it is not a
# real pile-of-law config (an earlier draft of this script assumed it
# was; it isn't in the dataset's own _DATA_URL mapping).
AVAILABLE_SUBSETS: Dict[str, str] = {
    "cfr": "train.cfr.jsonl.xz",
    "eurlex": "train.eurlex.jsonl.xz",
    "tos": "train.tos.jsonl.xz",
    "state_codes": "train.state_code.jsonl.xz",
    "uscode": "train.uscode.jsonl.xz",
    "echr": "train.echr.jsonl.xz",
    "federal_register": "train.federal_register.jsonl.xz",
    "tax_rulings": "train.taxrulings.jsonl.xz",
    "us_bills": "train.us_bills.jsonl.xz",
    "euro_parl": "train.euro_parl.jsonl.xz",
    "frcp": "train.frcp.jsonl.xz",
}

DEFAULT_SUBSETS = ["cfr", "eurlex", "tos"]

# Rough chars/token estimate.
# This is ONLY used as an early stopping approximation.
# Exact token counting is still performed later by chunk_text.py.
APPROX_CHARS_PER_TOKEN = 4

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


# ---------------------------------------------------------------------
# Streaming download
# ---------------------------------------------------------------------

def stream_jsonl_xz(
    url: str,
    output_file,
    max_docs: int,
    max_chars: int,
) -> Tuple[int, int]:
    n_docs = 0
    n_chars = 0

    print(f"  Source: {url}")

    last_error: Exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()

                # requests gives us compressed bytes.
                # We wrap the response stream with an XZ decompressor.
                with lzma.open(
                    response.raw,
                    mode="rt",
                    encoding="utf-8",
                    errors="replace",
                ) as xz_stream:

                    for line in xz_stream:
                        if not line.strip():
                            continue

                        try:
                            example = json.loads(line)
                        except json.JSONDecodeError:
                            print(
                                f"  Warning: skipping malformed JSON "
                                f"record at document {n_docs + 1}"
                            )
                            continue

                        text = example.get("text", "")
                        if not isinstance(text, str):
                            continue

                        text = text.strip()
                        if not text:
                            continue

                        output_file.write(text)
                        output_file.write("\n\n")

                        n_docs += 1
                        n_chars += len(text)

                        if n_docs >= max_docs or n_chars >= max_chars:
                            break

            return n_docs, n_chars

        except (requests.exceptions.RequestException, lzma.LZMAError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  All {MAX_RETRIES} attempts failed for {url}")

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts") from last_error


# ---------------------------------------------------------------------
# Download one subset
# ---------------------------------------------------------------------

def download_subset(
    subset: str,
    output_dir: str,
    max_docs: int,
    max_chars: int,
) -> Tuple[int, int]:

    if subset not in AVAILABLE_SUBSETS:
        raise ValueError(
            f"Unknown subset '{subset}'.\n\n"
            f"Available subsets:\n"
            f"  {', '.join(sorted(AVAILABLE_SUBSETS))}\n\n"
            f"Note: privacy_policies is NOT a current Pile-of-Law subset."
        )

    filename = AVAILABLE_SUBSETS[subset]
    url = BASE_URL + filename
    output_path = os.path.join(output_dir, f"{subset}.txt")

    print()
    print("=" * 70)
    print(f"Downloading subset: {subset}")
    print(f"Maximum documents: {max_docs:,}")
    print(
        f"Maximum characters: {max_chars:,} "
        f"(~{max_chars // APPROX_CHARS_PER_TOKEN:,} tokens)"
    )
    print("=" * 70)

    # Remove an old partial file so that a rerun doesn't accidentally
    # append duplicate documents.
    if os.path.exists(output_path):
        print(f"  Removing existing file: {output_path}")
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8") as output_file:
        n_docs, n_chars = stream_jsonl_xz(
            url=url,
            output_file=output_file,
            max_docs=max_docs,
            max_chars=max_chars,
        )

    print(f"  Documents written : {n_docs:,}")
    print(f"  Characters written: {n_chars:,}")
    print(f"  Output            : {output_path}")

    return n_docs, n_chars


# ---------------------------------------------------------------------
# Main corpus downloader
# ---------------------------------------------------------------------

def download_corpus(
    subsets: List[str],
    output_dir: str,
    max_docs_per_subset: int,
    max_tokens_per_subset: int,
) -> None:

    os.makedirs(output_dir, exist_ok=True)
    max_chars_per_subset = max_tokens_per_subset * APPROX_CHARS_PER_TOKEN

    total_docs = 0
    total_chars = 0

    print()
    print("Pile-of-Law DAPT downloader")
    print("-" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"Target tokens/subset: {max_tokens_per_subset:,}")
    print(f"Maximum documents/subset: {max_docs_per_subset:,}")
    print("-" * 70)

    for subset in subsets:
        n_docs, n_chars = download_subset(
            subset=subset,
            output_dir=output_dir,
            max_docs=max_docs_per_subset,
            max_chars=max_chars_per_subset,
        )
        total_docs += n_docs
        total_chars += n_chars

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Total documents : {total_docs:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Approx. tokens  : {total_chars // APPROX_CHARS_PER_TOKEN:,}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream legal text directly from Pile-of-Law Hugging Face files."
    )

    parser.add_argument(
        "--subsets", nargs="+", default=DEFAULT_SUBSETS,
        help=f"Pile-of-Law subsets to download. Available: {sorted(AVAILABLE_SUBSETS)}",
    )
    parser.add_argument(
        "--output_dir", default="data/raw/dapt",
        help="Directory where .txt files will be written.",
    )
    parser.add_argument(
        "--max_docs_per_subset", type=int, default=1500,
        help="Maximum number of documents per subset.",
    )
    parser.add_argument(
        "--max_tokens_per_subset", type=int, default=5_000_000,
        help="Approximate token limit per subset. Exact token count is "
             "performed later by chunk_text.py.",
    )

    args = parser.parse_args()

    download_corpus(
        subsets=args.subsets,
        output_dir=args.output_dir,
        max_docs_per_subset=args.max_docs_per_subset,
        max_tokens_per_subset=args.max_tokens_per_subset,
    )


if __name__ == "__main__":
    main()
