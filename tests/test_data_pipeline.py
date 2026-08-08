"""
Tests for the data pipeline scripts. These run entirely on CPU with a
small tokenizer and tiny fixtures — no GPU or Colab required, which is
exactly the point: catch data-pipeline bugs locally in VS Code before
burning Colab GPU time on them.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.scripts.build_sft_dataset import (
    REQUIRED_KEYS,
    build_dataset,
    record_to_pair,
    validate_record,
)
from data.scripts.chunk_text import chunk_corpus


# ---------------------------------------------------------------------------
# chunk_text.py
# ---------------------------------------------------------------------------

# A small, fast public tokenizer stands in for the real 1.5B model's
# tokenizer in tests — the chunking logic doesn't depend on model size.
TEST_TOKENIZER = "gpt2"


@pytest.fixture
def raw_corpus_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw")
        os.makedirs(raw_dir)
        with open(os.path.join(raw_dir, "doc1.txt"), "w") as f:
            f.write("The data controller shall implement appropriate technical measures. " * 40)
        with open(os.path.join(raw_dir, "doc2.txt"), "w") as f:
            f.write("Force majeure clauses excuse non-performance under statutory liability. " * 40)
        yield raw_dir


def test_chunk_corpus_produces_fixed_size_blocks(raw_corpus_dir):
    with tempfile.TemporaryDirectory() as out_dir:
        output_path = os.path.join(out_dir, "dapt_chunks.jsonl")
        n_blocks = chunk_corpus(
            input_dir=raw_corpus_dir,
            output_path=output_path,
            tokenizer_name=TEST_TOKENIZER,
            block_size=64,
            min_block_size=8,
        )
        assert n_blocks > 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == n_blocks
        for record in lines:
            assert "text" in record
            assert isinstance(record["text"], str)
            assert len(record["text"]) > 0


def test_chunk_corpus_raises_on_empty_dir():
    with tempfile.TemporaryDirectory() as empty_dir, tempfile.TemporaryDirectory() as out_dir:
        output_path = os.path.join(out_dir, "out.jsonl")
        with pytest.raises(FileNotFoundError):
            chunk_corpus(
                input_dir=empty_dir,
                output_path=output_path,
                tokenizer_name=TEST_TOKENIZER,
                block_size=64,
            )


# ---------------------------------------------------------------------------
# build_sft_dataset.py
# ---------------------------------------------------------------------------

VALID_RECORD = {
    "scenario": "A vendor retains customer data for 5 years without a documented basis.",
    "clause_violation": "GDPR Art. 5(1)(e) — storage limitation principle.",
    "recommended_remediation": "Define and enforce a data retention schedule tied to purpose.",
}


def test_validate_record_accepts_complete_record():
    assert validate_record(VALID_RECORD) is True


@pytest.mark.parametrize("missing_key", sorted(REQUIRED_KEYS))
def test_validate_record_rejects_missing_key(missing_key):
    incomplete = {k: v for k, v in VALID_RECORD.items() if k != missing_key}
    assert validate_record(incomplete) is False


def test_validate_record_rejects_empty_scenario():
    bad = dict(VALID_RECORD, scenario="   ")
    assert validate_record(bad) is False


def test_record_to_pair_formats_prompt_and_response():
    pair = record_to_pair(VALID_RECORD)
    assert "prompt" in pair and "response" in pair
    assert VALID_RECORD["scenario"] in pair["prompt"]
    assert VALID_RECORD["clause_violation"] in pair["response"]
    assert VALID_RECORD["recommended_remediation"] in pair["response"]


def test_build_dataset_writes_jsonl_and_skips_invalid():
    with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
        with open(os.path.join(in_dir, "batch1.json"), "w") as f:
            json.dump([VALID_RECORD, {"scenario": "incomplete only"}], f)

        output_path = os.path.join(out_dir, "sft_pairs.jsonl")
        n_written = build_dataset(in_dir, output_path)

        assert n_written == 1
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["prompt"].startswith("You are a legal compliance auditor")
