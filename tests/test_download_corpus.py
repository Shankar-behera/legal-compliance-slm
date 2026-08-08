"""
Tests for data/scripts/download_corpus.py — subset validation and both
stopping criteria (max_docs, max_chars). `datasets.load_dataset` is
stubbed/monkeypatched so these run without network access.
"""
import os
import sys
import tempfile
import types

if "datasets" not in sys.modules:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = fake_datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import data.scripts.download_corpus as download_corpus_mod
from data.scripts.download_corpus import download_corpus, download_subset


def _fake_docs(n: int):
    return [
        {"text": f"Section {i}. This is a compliance clause about data handling."}
        for i in range(n)
    ]


def test_download_subset_rejects_unknown_subset():
    with pytest.raises(ValueError):
        download_subset("not_a_real_subset", "/tmp", max_docs=10, max_chars=1000)


def test_download_subset_stops_at_max_docs():
    download_corpus_mod.load_dataset = (
        lambda name, subset, split, streaming, trust_remote_code: iter(_fake_docs(50))
    )
    with tempfile.TemporaryDirectory() as out_dir:
        n = download_subset("cfr", out_dir, max_docs=10, max_chars=10_000_000)
        assert n == 10

        out_path = os.path.join(out_dir, "cfr.txt")
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "Section 0." in content
        assert "Section 9." in content
        assert "Section 10." not in content


def test_download_subset_stops_at_max_chars():
    download_corpus_mod.load_dataset = (
        lambda name, subset, split, streaming, trust_remote_code: iter(_fake_docs(50))
    )
    with tempfile.TemporaryDirectory() as out_dir:
        n = download_subset("cfr", out_dir, max_docs=1000, max_chars=200)
        assert 0 < n < 50


def test_download_corpus_writes_one_file_per_subset():
    download_corpus_mod.load_dataset = (
        lambda name, subset, split, streaming, trust_remote_code: iter(_fake_docs(5))
    )
    with tempfile.TemporaryDirectory() as out_dir:
        download_corpus(
            subsets=["cfr", "eurlex"],
            output_dir=out_dir,
            max_docs_per_subset=5,
            max_tokens_per_subset=1_000_000,
        )
        assert sorted(os.listdir(out_dir)) == ["cfr.txt", "eurlex.txt"]
