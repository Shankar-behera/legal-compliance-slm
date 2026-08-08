"""
Plots loss (train vs. eval, where available) against training steps, read
directly from the `trainer_state.json` that HF Trainer writes into every
checkpoint directory — no need for W&B access to reproduce the curve for
the README.

Usage:
    python -m src.training.plot_curves \
        --trainer_state_path /content/drive/.../checkpoints/sft/final_adapter/trainer_state.json \
        --output_path docs/training_curve_sft.png \
        --title "SFT Loss (Qwen2.5-1.5B + QLoRA)"
"""
import argparse
import json
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe for Colab / CI
import matplotlib.pyplot as plt


def load_log_history(trainer_state_path: str) -> List[dict]:
    with open(trainer_state_path, "r") as f:
        state = json.load(f)
    return state.get("log_history", [])


def extract_series(log_history: List[dict]) -> Tuple[List[int], List[float], List[int], List[float]]:
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    return train_steps, train_losses, eval_steps, eval_losses


def plot_curve(trainer_state_path: str, output_path: str, title: str = "Training Loss") -> str:
    log_history = load_log_history(trainer_state_path)
    train_steps, train_losses, eval_steps, eval_losses = extract_series(log_history)

    if not train_steps and not eval_steps:
        raise ValueError(
            f"No loss entries found in {trainer_state_path}. "
            f"Check that logging_steps was set and training actually ran."
        )

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train loss", linewidth=1.5)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="eval loss", linewidth=1.5, linestyle="--")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer_state_path", required=True)
    parser.add_argument("--output_path", default="docs/training_curve.png")
    parser.add_argument("--title", default="Training Loss")
    args = parser.parse_args()

    path = plot_curve(args.trainer_state_path, args.output_path, args.title)
    print(f"Saved curve to {path}")


if __name__ == "__main__":
    main()
