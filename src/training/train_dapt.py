"""
Phase 2 — Domain-Adaptive Continued Pre-Training.

Loads the base model in 4-bit, continues causal-LM pretraining on the
chunked legal corpus, and checkpoints to a Drive-mounted output_dir.

Usage (inside Colab, after `!pip install -r requirements.txt`):
    from src.training.train_dapt import run_dapt
    run_dapt("configs/dapt_config.yaml")

Or from the CLI:
    python -m src.training.train_dapt --config configs/dapt_config.yaml
"""
import argparse

import torch
import yaml

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.dapt_dataset import load_dapt_dataset
from src.training.callbacks import DriveCheckpointGuard, LossSpikeMonitor


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_quant_config(cfg: dict) -> BitsAndBytesConfig:
    """Build 4-bit quantization configuration."""
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(
            torch,
            q["bnb_4bit_compute_dtype"]
        ),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def build_lora_config(lora_config_path: str) -> LoraConfig:
    """Build LoRA configuration from YAML file."""
    with open(lora_config_path, "r") as f:
        lora_cfg = yaml.safe_load(f)
    
    return LoraConfig(**lora_cfg)


def run_dapt(config_path: str) -> str:
    """Run Domain-Adaptive Pre-Training with LoRA."""
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    quant_config = build_quant_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quant_config,
        device_map="auto",
    )
    
    model.config.use_cache = False  # required alongside gradient checkpointing

    # Prepare the 4-bit model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Load the project's shared LoRA configuration
    lora_config = build_lora_config(cfg["lora_config"])

    # Attach trainable LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dapt_dataset(
        processed_path=cfg["data"]["processed_path"],
        tokenizer=tokenizer,
        block_size=cfg["data"]["block_size"],
        text_column=cfg["data"]["text_column"],
    )

    # Training arguments
    t = cfg["training"]
    training_args = TrainingArguments(
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
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[DriveCheckpointGuard(), LossSpikeMonitor()],
    )

    # Train and save
    # Train and resume from the latest checkpoint if one exists
    last_checkpoint = get_last_checkpoint(cfg["output_dir"])

    if last_checkpoint is not None:
        print(f"Resuming DAPT from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Starting DAPT from scratch.")
        trainer.train()

    final_dir = f"{cfg['output_dir']}/final_adapter"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(
        f"DAPT complete. "
        f"Domain-adapted LoRA adapter saved to {final_dir}"
    )
    return final_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dapt_config.yaml")
    args = parser.parse_args()
    run_dapt(args.config)


if __name__ == "__main__":
    main()