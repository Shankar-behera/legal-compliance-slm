"""
Phase 3 — QLoRA Supervised Fine-Tuning on compliance instruction pairs.

Loads the DAPT checkpoint in 4-bit, attaches a LoRA adapter, and trains
with TRL's SFTTrainer so the model learns compliance-auditor *behavior*
on top of the domain vocabulary DAPT already injected.

Usage:
    python -m src.training.train_sft --config configs/sft_config.yaml
"""
import argparse

import torch
import yaml
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from src.data.sft_dataset import load_sft_dataset
from src.training.callbacks import DriveCheckpointGuard, LossSpikeMonitor


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_quant_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def build_lora_config(lora_config_path: str) -> LoraConfig:
    with open(lora_config_path, "r") as f:
        lora_cfg = yaml.safe_load(f)
    return LoraConfig(**lora_cfg)


def run_sft(config_path: str) -> str:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quant_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_path"],
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(cfg["lora_config"])

    dataset = load_sft_dataset(
        processed_path=cfg["data"]["processed_path"],
        eval_split_ratio=cfg["eval"]["eval_split_ratio"],
        seed=cfg.get("seed", 42),
    )

    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        eval_strategy="steps",
        eval_steps=cfg["eval"]["eval_steps"],
        max_seq_length=cfg["data"]["max_seq_length"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        callbacks=[DriveCheckpointGuard(), LossSpikeMonitor()],
    )

    trainer.train()

    final_dir = f"{cfg['output_dir']}/final_adapter"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"SFT complete. LoRA adapter saved to {final_dir}")
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.yaml")
    args = parser.parse_args()
    run_sft(args.config)


if __name__ == "__main__":
    main()
