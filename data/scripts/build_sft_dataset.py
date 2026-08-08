"""
Builds the instruction-tuning dataset for Phase 3 (QLoRA SFT).

Input: a directory of raw scenario JSON files, each shaped as
    {
        "scenario": "...",
        "clause_violation": "...",
        "recommended_remediation": "..."
    }
(this is the format to target if you're hand-writing or LLM-assisting
synthetic compliance examples; adapt `record_to_pair` if you're pulling
from LexGLUE or another external dataset instead.)

Output: data/processed/sft_pairs.jsonl with {"prompt": ..., "response": ...}
records ready for TRL's SFTTrainer.

Usage:
    python data/scripts/build_sft_dataset.py \
        --input_dir data/raw/sft \
        --output_path data/processed/sft_pairs.jsonl
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator

PROMPT_TEMPLATE = (
    "You are a legal compliance auditor. Review the following scenario, "
    "identify any clause violations, and recommend remediation.\n\n"
    "Scenario:\n{scenario}\n\nAudit:"
)

RESPONSE_TEMPLATE = (
    "Clause Violation: {clause_violation}\n"
    "Recommended Remediation: {recommended_remediation}"
)

REQUIRED_KEYS = {"scenario", "clause_violation", "recommended_remediation"}


def iter_raw_records(input_dir: str) -> Iterator[Dict]:
    paths = sorted(Path(input_dir).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json scenario files found under {input_dir}")
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # a file may contain a single record or a list of records
        records = data if isinstance(data, list) else [data]
        for record in records:
            yield record


def validate_record(record: Dict, source: str = "") -> bool:
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        print(f"[skip] record missing keys {missing} ({source})")
        return False
    if not record["scenario"].strip():
        print(f"[skip] empty scenario ({source})")
        return False
    return True


def record_to_pair(record: Dict) -> Dict[str, str]:
    prompt = PROMPT_TEMPLATE.format(scenario=record["scenario"].strip())
    response = RESPONSE_TEMPLATE.format(
        clause_violation=record["clause_violation"].strip(),
        recommended_remediation=record["recommended_remediation"].strip(),
    )
    return {"prompt": prompt, "response": response}


def build_dataset(input_dir: str, output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in iter_raw_records(input_dir):
            if not validate_record(record):
                continue
            pair = record_to_pair(record)
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    n_written = build_dataset(args.input_dir, args.output_path)
    print(f"Wrote {n_written} instruction pairs to {args.output_path}")
    if n_written < 3000:
        print(
            f"Warning: {n_written} pairs is below the ~3,000-5,000 target "
            f"volume for stable SFT convergence on a 1.5B model."
        )


if __name__ == "__main__":
    main()
