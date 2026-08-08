"""
Pulls the `unfair_tos` config of LexGLUE (lex_glue) and reformats it into
this project's raw SFT record schema —
    {"scenario": ..., "clause_violation": ..., "recommended_remediation": ...}
— so it drops straight into `data/raw/sft/` and gets picked up by
`build_sft_dataset.py` alongside any synthetic scenario files.

Why unfair_tos and not another LexGLUE config: it's the only LexGLUE
subset that's actually clause-level violation detection (unfair Terms of
Service clauses across 8 categories) rather than document classification
or outcome prediction — the others (ecthr_a/b, scotus, eurlex, ledgar,
case_hold) don't map onto "scenario → clause violation → remediation"
without changing what the task fundamentally is.

Multi-label caveat: a ToS sentence can carry more than one unfairness
label. Each label produces its own record (the audit format is
one-violation-at-a-time), so total record count is >= number of examples
with at least one label. `--include_fair_examples` optionally emits a
matching number of "no violation identified" negatives from unlabeled
clauses, which is worth including so the model doesn't learn to always
flag something.

Usage:
    python data/scripts/build_sft_from_lexglue.py \
        --output_path data/raw/sft/lexglue_scenarios.json \
        --max_examples 4000 \
        --include_fair_examples
"""
import argparse
import json
import random
from typing import Dict, List, Optional

from datasets import load_dataset

# LexGLUE unfair_tos label id -> (category name, remediation template).
# Remediation text is deterministic/templated, not LLM-generated — keeps
# this script reproducible and free of an API dependency. If you want
# richer, more varied remediation phrasing, feed these through the
# synthetic-generation script as a rewriting pass instead of editing here.
LABEL_MAP: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Limitation of liability",
        "remediation": (
            "Revise the liability clause to remove blanket exclusions and cap "
            "liability at a level proportionate to the service, consistent with "
            "applicable consumer protection law rather than disclaiming all liability."
        ),
    },
    1: {
        "name": "Unilateral termination",
        "remediation": (
            "Amend the termination clause to require reasonable notice and a "
            "stated basis for termination, rather than allowing termination "
            "at the provider's sole discretion without cause or notice."
        ),
    },
    2: {
        "name": "Unilateral change",
        "remediation": (
            "Require advance notice and, where the change is material, user "
            "consent before contract terms take effect, rather than allowing "
            "silent unilateral modification."
        ),
    },
    3: {
        "name": "Content removal",
        "remediation": (
            "Specify clear, objective grounds and a notice process for content "
            "removal, rather than reserving unrestricted discretionary removal rights."
        ),
    },
    4: {
        "name": "Contract by using",
        "remediation": (
            "Replace implied acceptance via continued use with an explicit, "
            "affirmative consent mechanism (e.g. a checkbox or signature) for "
            "the terms to bind the user."
        ),
    },
    5: {
        "name": "Choice of law",
        "remediation": (
            "Ensure the choice-of-law clause does not deprive the user of "
            "mandatory consumer protections available under their home "
            "jurisdiction's law."
        ),
    },
    6: {
        "name": "Jurisdiction",
        "remediation": (
            "Avoid mandating a forum that is unreasonably inconvenient or "
            "inaccessible to the user; align the jurisdiction clause with "
            "applicable consumer-protection venue rules."
        ),
    },
    7: {
        "name": "Arbitration",
        "remediation": (
            "Ensure any mandatory arbitration clause preserves the user's "
            "right to pursue claims in small-claims court and does not waive "
            "class-action rights where such waivers are unenforceable."
        ),
    },
}

NO_VIOLATION_REMEDIATION = (
    "No unfair clause pattern identified in this text; no remediation required."
)


def record_for_labeled_example(text: str, label_id: int) -> Optional[Dict[str, str]]:
    label_info = LABEL_MAP.get(label_id)
    if label_info is None:
        return None
    return {
        "scenario": text.strip(),
        "clause_violation": f"Unfair ToS clause — {label_info['name']}",
        "recommended_remediation": label_info["remediation"],
    }


def record_for_fair_example(text: str) -> Dict[str, str]:
    return {
        "scenario": text.strip(),
        "clause_violation": "None identified",
        "recommended_remediation": NO_VIOLATION_REMEDIATION,
    }


def build_records(
    max_examples: Optional[int] = None,
    include_fair_examples: bool = False,
    seed: int = 42,
) -> List[Dict[str, str]]:
    dataset = load_dataset("lex_glue", "unfair_tos", split="train")

    violation_records: List[Dict[str, str]] = []
    fair_texts: List[str] = []

    for example in dataset:
        text = example["text"]
        labels = example["labels"]
        if not text or not text.strip():
            continue

        if labels:
            for label_id in labels:
                record = record_for_labeled_example(text, label_id)
                if record is not None:
                    violation_records.append(record)
        else:
            fair_texts.append(text)

    rng = random.Random(seed)
    rng.shuffle(violation_records)
    if max_examples is not None:
        violation_records = violation_records[:max_examples]

    records = list(violation_records)

    if include_fair_examples:
        rng.shuffle(fair_texts)
        n_fair = len(violation_records) if max_examples is None else min(
            len(fair_texts), max_examples // 4  # keep negatives a minority, not half
        )
        records.extend(record_for_fair_example(t) for t in fair_texts[:n_fair])
        rng.shuffle(records)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", default="data/raw/sft/lexglue_scenarios.json")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--include_fair_examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_records(
        max_examples=args.max_examples,
        include_fair_examples=args.include_fair_examples,
        seed=args.seed,
    )

    import os

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_violation = sum(1 for r in records if r["clause_violation"] != "None identified")
    n_fair = len(records) - n_violation
    print(
        f"Wrote {len(records)} records to {args.output_path} "
        f"({n_violation} violation examples, {n_fair} 'no violation' negatives)"
    )


if __name__ == "__main__":
    main()
