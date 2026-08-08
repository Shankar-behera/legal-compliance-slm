"""
Local inference wrapper around the llama.cpp CLI binary, for running the
exported GGUF model without pulling in the full Python/HF stack again.

Usage:
    python inference/run_llamacpp.py \
        --model_path models/gguf/legal-slm.Q4_K_M.gguf \
        --prompt "A vendor stores customer data for 7 years with no stated basis."
"""
import argparse
import subprocess
from typing import List

DEFAULT_LLAMA_CLI = "llama.cpp/build/bin/llama-cli"

STOP_MARKER = "[end of text]"


def build_llamacpp_command(
    model_path: str,
    prompt: str,
    n_predict: int = 256,
    temperature: float = 0.2,
    llama_cli_path: str = DEFAULT_LLAMA_CLI,
) -> List[str]:
    if not model_path:
        raise ValueError("model_path is required")
    return [
        llama_cli_path,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        str(temperature),
    ]


def parse_llamacpp_output(raw_output: str, prompt: str) -> str:
    text = raw_output.strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    if STOP_MARKER in text:
        text = text.split(STOP_MARKER)[0].strip()
    return text


def run_inference(
    model_path: str,
    prompt: str,
    n_predict: int = 256,
    temperature: float = 0.2,
    llama_cli_path: str = DEFAULT_LLAMA_CLI,
) -> str:
    cmd = build_llamacpp_command(model_path, prompt, n_predict, temperature, llama_cli_path)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_llamacpp_output(result.stdout, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n_predict", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llama_cli_path", default=DEFAULT_LLAMA_CLI)
    args = parser.parse_args()

    response = run_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        n_predict=args.n_predict,
        temperature=args.temperature,
        llama_cli_path=args.llama_cli_path,
    )
    print(response)


if __name__ == "__main__":
    main()
