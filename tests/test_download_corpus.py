"""
Tests for data/scripts/download_corpus.py — subset validation, both
stopping criteria (max_docs, max_chars), retry-on-failure behavior, and
that the config->filename map matches pile-of-law's own loading script.

`requests.get` is monkeypatched with a fake context-manager response
whose `.raw` is a real lzma-compressed JSONL payload, so the actual
decompression + parsing path is exercised end-to-end without any
network access.
"""
import io
import json
import lzma
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.scripts.download_corpus as dl_mod
from data.scripts.download_corpus import (
    AVAILABLE_SUBSETS,
    download_corpus,
    download_subset,
)


def _fake_xz_payload(n: int) -> bytes:
    lines = [
        json.dumps({"text": f"Section {i}. Compliance clause about data handling."})
        for i in range(n)
    ]
    return lzma.compress("\n".join(lines).encode("utf-8"))


class _FakeResponse:
    """Mimics requests.Response as used by stream_jsonl_xz: a context
    manager exposing .raise_for_status() and a .raw file-like object."""

    def __init__(self, payload: bytes, status_error: Exception = None):
        self.raw = io.BytesIO(payload)
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_requests_get(payload: bytes):
    def fake_get(url, stream=True, timeout=120):
        return _FakeResponse(payload)

    dl_mod.requests.get = fake_get


def test_available_subsets_match_pile_of_law_loading_script():
    # Spot-check against pile-of-law.py's own _DATA_URL mapping — these
    # are not guesses, they're copied from the dataset's source.
    assert AVAILABLE_SUBSETS["cfr"] == "train.cfr.jsonl.xz"
    assert AVAILABLE_SUBSETS["state_codes"] == "train.state_code.jsonl.xz"  # singular in filename
    assert AVAILABLE_SUBSETS["tax_rulings"] == "train.taxrulings.jsonl.xz"  # no underscore
    assert "privacy_policies" not in AVAILABLE_SUBSETS  # not a real pile-of-law config


def test_download_subset_rejects_unknown_subset():
    with pytest.raises(ValueError):
        download_subset("not_a_real_subset", "/tmp", max_docs=10, max_chars=1000)


def test_download_subset_stops_at_max_docs():
    _patch_requests_get(_fake_xz_payload(50))
    with tempfile.TemporaryDirectory() as out_dir:
        n_docs, n_chars = download_subset("cfr", out_dir, max_docs=10, max_chars=10_000_000)
        assert n_docs == 10

        out_path = os.path.join(out_dir, "cfr.txt")
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "Section 0." in content
        assert "Section 9." in content
        assert "Section 10." not in content


def test_download_subset_stops_at_max_chars():
    _patch_requests_get(_fake_xz_payload(50))
    with tempfile.TemporaryDirectory() as out_dir:
        n_docs, _ = download_subset("cfr", out_dir, max_docs=1000, max_chars=200)
        assert 0 < n_docs < 50


def test_download_subset_removes_stale_partial_file():
    with tempfile.TemporaryDirectory() as out_dir:
        stale_path = os.path.join(out_dir, "cfr.txt")
        with open(stale_path, "w") as f:
            f.write("leftover partial content from a previous failed run\n")

        _patch_requests_get(_fake_xz_payload(5))
        download_subset("cfr", out_dir, max_docs=100, max_chars=10_000_000)

        with open(stale_path) as f:
            content = f.read()
        assert "leftover partial content" not in content


def test_download_corpus_writes_one_file_per_subset():
    _patch_requests_get(_fake_xz_payload(5))
    with tempfile.TemporaryDirectory() as out_dir:
        download_corpus(
            subsets=["cfr", "eurlex"],
            output_dir=out_dir,
            max_docs_per_subset=5,
            max_tokens_per_subset=1_000_000,
        )
        assert sorted(os.listdir(out_dir)) == ["cfr.txt", "eurlex.txt"]


def test_stream_jsonl_xz_retries_then_succeeds():
    import requests

    payload = _fake_xz_payload(5)
    calls = {"n": 0}

    def flaky_get(url, stream=True, timeout=120):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.ConnectionError("simulated network blip")
        return _FakeResponse(payload)

    dl_mod.requests.get = flaky_get
    dl_mod.RETRY_BACKOFF_SECONDS = 0  # don't actually sleep in tests

    with tempfile.TemporaryDirectory() as out_dir:
        out_path = os.path.join(out_dir, "cfr.txt")
        with open(out_path, "w") as f:
            n_docs, n_chars = dl_mod.stream_jsonl_xz(
                url="https://example.com/fake.jsonl.xz",
                output_file=f,
                max_docs=100,
                max_chars=10_000_000,
            )

    assert calls["n"] == 2
    assert n_docs == 5


def test_stream_jsonl_xz_raises_after_exhausting_retries():
    import requests

    def always_fails(url, stream=True, timeout=120):
        raise requests.exceptions.ConnectionError("simulated persistent failure")

    dl_mod.requests.get = always_fails
    dl_mod.RETRY_BACKOFF_SECONDS = 0

    with tempfile.TemporaryDirectory() as out_dir:
        out_path = os.path.join(out_dir, "cfr.txt")
        with open(out_path, "w") as f:
            with pytest.raises(RuntimeError):
                dl_mod.stream_jsonl_xz(
                    url="https://example.com/fake.jsonl.xz",
                    output_file=f,
                    max_docs=100,
                    max_chars=10_000_000,
                )


def test_stream_jsonl_xz_skips_malformed_json_lines():
    good_line = json.dumps({"text": "A valid compliance clause about retention."})
    payload = lzma.compress(f"{good_line}\nnot valid json\n{good_line}\n".encode("utf-8"))
    _patch_requests_get(payload)

    with tempfile.TemporaryDirectory() as out_dir:
        n_docs, _ = download_subset("cfr", out_dir, max_docs=100, max_chars=10_000_000)
        assert n_docs == 2  # both good lines written, malformed line skipped
