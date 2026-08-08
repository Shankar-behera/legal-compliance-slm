"""
Phase 4b — Convert the merged HF model to GGUF and quantize it for local
CPU/GPU inference via llama.cpp or Ollama.

Assumes llama.cpp has been cloned and built, e.g. inside the Colab
notebook:
    !git clone https://github.com/ggerganov/llama.cpp
    !pip install -r llama.cpp/requirements.txt -q
    !cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build --config Release -j

Usage:
    python -m src.export.convert_gguf \
        --merged_model_path /content/drive/MyDrive/legal-compliance-slm/merged \
        --llama_cpp_dir llama.cpp \
        --output_dir models/gguf \
        --quant_type Q4_K_M
"""
import argparse
import os
import subprocess
import sys
from typing import List


VALID_QUANT_TYPES = {"Q4_K_M", "Q5_K_M", "Q8_0", "F16"}


def build_convert_command(
    merged_model_path: str, llama_cpp_dir: str, fp16_output_path: str
) -> List[str]:
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    return [
        sys.executable,
        convert_script,
        merged_model_path,
        "--outfile",
        fp16_output_path,
        "--outtype",
        "f16",
    ]


def build_quantize_command(
    llama_cpp_dir: str, fp16_path: str, quantized_path: str, quant_type: str
) -> List[str]:
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    return [quantize_bin, fp16_path, quantized_path, quant_type]


def convert_to_gguf(
    merged_model_path: str,
    llama_cpp_dir: str,
    output_dir: str,
    quant_type: str = "Q4_K_M",
) -> str:
    if quant_type not in VALID_QUANT_TYPES:
        raise ValueError(f"quant_type must be one of {VALID_QUANT_TYPES}, got {quant_type}")

    os.makedirs(output_dir, exist_ok=True)
    fp16_path = os.path.join(output_dir, "legal-slm.f16.gguf")
    quantized_path = os.path.join(output_dir, f"legal-slm.{quant_type}.gguf")

    convert_cmd = build_convert_command(merged_model_path, llama_cpp_dir, fp16_path)
    print("Running:", " ".join(convert_cmd))
    subprocess.run(convert_cmd, check=True)

    quantize_cmd = build_quantize_command(llama_cpp_dir, fp16_path, quantized_path, quant_type)
    print("Running:", " ".join(quantize_cmd))
    subprocess.run(quantize_cmd, check=True)

    print(f"Quantized GGUF ready at {quantized_path}")
    return quantized_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_model_path", required=True)
    parser.add_argument("--llama_cpp_dir", default="llama.cpp")
    parser.add_argument("--output_dir", default="models/gguf")
    parser.add_argument("--quant_type", default="Q4_K_M", choices=sorted(VALID_QUANT_TYPES))
    args = parser.parse_args()

    convert_to_gguf(
        merged_model_path=args.merged_model_path,
        llama_cpp_dir=args.llama_cpp_dir,
        output_dir=args.output_dir,
        quant_type=args.quant_type,
    )


if __name__ == "__main__":
    main()
