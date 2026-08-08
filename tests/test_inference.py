"""
Tests for the inference wrapper (inference/run_llamacpp.py).

These are structural/contract tests — they check that the wrapper builds
correct commands and parses output correctly. They deliberately do NOT
require a real GGUF file or a GPU, so they run in plain VS Code/CI.
Actual model-quality evaluation belongs in src/eval/, run after a real
training run, not here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.run_llamacpp import build_llamacpp_command, parse_llamacpp_output


def test_build_llamacpp_command_includes_required_flags():
    cmd = build_llamacpp_command(
        model_path="models/gguf/legal-slm.Q4_K_M.gguf",
        prompt="Audit this contract clause.",
        n_predict=256,
        temperature=0.2,
    )
    assert "models/gguf/legal-slm.Q4_K_M.gguf" in cmd
    assert "-p" in cmd
    assert "Audit this contract clause." in cmd
    assert "--temp" in cmd
    assert "0.2" in cmd
    assert "-n" in cmd
    assert "256" in cmd


def test_build_llamacpp_command_rejects_missing_model_path():
    with pytest.raises(ValueError):
        build_llamacpp_command(model_path="", prompt="test", n_predict=64, temperature=0.2)


def test_parse_llamacpp_output_strips_prompt_echo():
    prompt = "Audit this contract clause."
    raw_output = f"{prompt} The clause violates GDPR Art. 5.\n\n[end of text]"
    parsed = parse_llamacpp_output(raw_output, prompt)
    assert parsed == "The clause violates GDPR Art. 5."


def test_parse_llamacpp_output_handles_no_echo():
    raw_output = "The clause violates GDPR Art. 5.\n\n[end of text]"
    parsed = parse_llamacpp_output(raw_output, prompt="Audit this contract clause.")
    assert parsed == "The clause violates GDPR Art. 5."
