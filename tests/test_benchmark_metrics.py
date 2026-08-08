"""
Tests for the pure scoring functions in src/eval/benchmark.py — exact
match, F1, and markdown table formatting. These don't touch a model or
tokenizer, so they run anywhere (no GPU, no network).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.benchmark import (
    BenchmarkRow,
    compute_exact_match,
    compute_f1,
    rows_to_markdown_table,
)


def test_exact_match_identical_strings():
    assert compute_exact_match("Clause 5 violated.", "Clause 5 violated.") == 1.0


def test_exact_match_case_and_whitespace_insensitive():
    assert compute_exact_match("  Clause 5  violated. ", "clause 5 violated.") == 1.0


def test_exact_match_different_strings():
    assert compute_exact_match("Clause 5 violated.", "Clause 7 violated.") == 0.0


def test_f1_identical_strings_is_one():
    assert compute_f1("data controller shall notify", "data controller shall notify") == 1.0


def test_f1_partial_overlap():
    score = compute_f1("data controller shall notify authority", "data controller must notify")
    assert 0.0 < score < 1.0


def test_f1_no_overlap_is_zero():
    assert compute_f1("completely different words here", "totally unrelated text now") == 0.0


def test_f1_handles_empty_strings():
    assert compute_f1("", "something") == 0.0
    assert compute_f1("something", "") == 0.0


def test_rows_to_markdown_table_formats_all_rows():
    rows = [
        BenchmarkRow("Base model", "qwen-base", 12.3, 0.1, 0.35, 0.4, 210.5),
        BenchmarkRow("After DAPT", "qwen-dapt", 9.8, 0.15, 0.42, 0.3, 215.0),
        BenchmarkRow("After SFT", "qwen-sft", 6.1, 0.55, 0.78, 0.08, 220.2),
    ]
    table = rows_to_markdown_table(rows)

    assert "Base model" in table
    assert "After DAPT" in table
    assert "After SFT" in table
    assert "Perplexity" in table
    assert "Hallucination Rate" in table
    # header + separator + 3 data rows
    assert len(table.strip().split("\n")) == 5
