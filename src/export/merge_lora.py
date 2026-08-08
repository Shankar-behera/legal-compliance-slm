"""
Phase 4a — Merge the trained LoRA adapter into the DAPT base model to
produce a single standalone set of full-precision weights, ready for
GGUF conversion.

Usage:
    python -m src.export.merge_lora \
        --base_model_path /content/drive/MyDrive/legal-compliance-slm/checkpoints/dapt/final \
        --adapter_path /content/drive/MyDrive/legal-compliance-slm/checkpoints/sft/final_adapter \
        --output_path /content/drive/MyDrive/legal-compliance-slm/merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_model_path: str, adapter_path: str, output_path: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load in full precision (not 4-bit) — merging requires dequantized weights.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    merge_lora(args.base_model_path, args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()
