"""Dataset loading for QLoRA instruction fine-tuning."""
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset


def load_sft_dataset(
    processed_path: str,
    prompt_column: str = "prompt",
    response_column: str = "response",
    eval_split_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load instruction pairs (produced by data/scripts/build_sft_dataset.py)
    and format them into a single `text` field TRL's SFTTrainer can consume
    directly, using the base tokenizer's chat template convention.
    """
    raw = load_dataset("json", data_files=processed_path, split="train")

    def format_fn(example: Dict) -> Dict:
        example["text"] = (
            f"{example[prompt_column]}\n{example[response_column]}"
        )
        return example

    formatted = raw.map(format_fn, desc="Formatting SFT pairs")
    formatted = formatted.remove_columns(
        [c for c in [prompt_column, response_column] if c in formatted.column_names]
    )

    split = formatted.train_test_split(test_size=eval_split_ratio, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])
