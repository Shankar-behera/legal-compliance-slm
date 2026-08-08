"""
Tests for data/scripts/build_sft_from_lexglue.py — label mapping,
multi-label expansion, and fair/violation splitting. `datasets.load_dataset`
is stubbed so these run without network access; the real pull is exercised
manually (see docs/dataset_stats.md workflow) once network is available.
"""
import os
import sys
import types

# Stub `datasets` before importing the module under test, so the module's
# top-level `from datasets import load_dataset` succeeds without the real
# package installed.
if "datasets" not in sys.modules:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = fake_datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.scripts.build_sft_from_lexglue as lexglue_mod
from data.scripts.build_sft_from_lexglue import (
    LABEL_MAP,
    NO_VIOLATION_REMEDIATION,
    record_for_fair_example,
    record_for_labeled_example,
)


def test_label_map_covers_all_eight_unfair_tos_categories():
    assert len(LABEL_MAP) == 8
    for label_id in range(8):
        assert label_id in LABEL_MAP
        assert LABEL_MAP[label_id]["name"]
        assert LABEL_MAP[label_id]["remediation"]


def test_record_for_labeled_example_shapes_correctly():
    record = record_for_labeled_example(
        "The Company may terminate this agreement at any time without notice.", 1
    )
    assert record["scenario"].startswith("The Company may terminate")
    assert "Unilateral termination" in record["clause_violation"]
    assert "notice" in record["recommended_remediation"]


def test_record_for_labeled_example_unknown_id_returns_none():
    assert record_for_labeled_example("some text", 99) is None


def test_record_for_fair_example_uses_no_violation_template():
    record = record_for_fair_example("This is a balanced clause.")
    assert record["clause_violation"] == "None identified"
    assert record["recommended_remediation"] == NO_VIOLATION_REMEDIATION


def test_build_records_expands_multi_label_examples():
    fake_examples = [
        {"text": "Provider may change these terms at any time.", "labels": [2]},
        {"text": "Liability is capped reasonably per statute.", "labels": []},
        {"text": "By using this site you agree to arbitration and terms.", "labels": [4, 7]},
    ]
    lexglue_mod.load_dataset = lambda name, config, split: fake_examples

    records = lexglue_mod.build_records(max_examples=None, include_fair_examples=False)

    # one single-label example (1 record) + one multi-label example (2 records) = 3
    assert len(records) == 3
    violations = {r["clause_violation"] for r in records}
    assert "Unfair ToS clause — Unilateral change" in violations
    assert "Unfair ToS clause — Contract by using" in violations
    assert "Unfair ToS clause — Arbitration" in violations


def test_build_records_includes_fair_negatives_when_requested():
    fake_examples = [
        {"text": "Provider may change these terms at any time.", "labels": [2]},
        {"text": "This is a normal, balanced clause.", "labels": []},
    ]
    lexglue_mod.load_dataset = lambda name, config, split: fake_examples

    records = lexglue_mod.build_records(max_examples=None, include_fair_examples=True)
    fair_records = [r for r in records if r["clause_violation"] == "None identified"]
    assert len(fair_records) >= 1
