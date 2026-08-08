"""Dataset loading for Domain-Adaptive Continued Pre-Training."""
from typing import Dict

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def load_dapt_dataset(
    processed_path: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 512,
    text_column: str = "text",
    streaming: bool = False,
) -> Dataset:
    """
    Load the pre-chunked DAPT JSONL (produced by data/scripts/chunk_text.py)
    and tokenize it for causal LM training. Chunks are already fixed-length
    at the character/token-boundary level from the chunker, so this just
    re-tokenizes to tensors and sets labels = input_ids for CLM.
    """
    raw = load_dataset("json", data_files=processed_path, split="train", streaming=streaming)

    def tokenize_fn(examples: Dict) -> Dict:
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    if streaming:
        return raw.map(tokenize_fn, batched=True, remove_columns=[text_column])

    tokenized_ds = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,
        desc="Tokenizing DAPT corpus",
    )
    return tokenized_ds
