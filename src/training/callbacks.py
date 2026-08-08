"""Custom Trainer callbacks — checkpoint safety and lightweight logging."""
import json
import os
import time
from typing import Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class DriveCheckpointGuard(TrainerCallback):
    """
    Belt-and-suspenders checkpoint safety for free-tier Colab.

    HF Trainer already writes checkpoints to `output_dir` (which should
    point at a Drive-mounted path) on `save_steps`. This callback adds:
      1. A lightweight `run_state.json` written alongside each checkpoint,
         recording step/epoch/loss/wall-clock time, so a resumed run can
         be verified quickly without loading the full trainer state.
      2. A warning if the gap between saves exceeds `max_save_gap_seconds`,
         which usually means something upstream (e.g. data loading) is
         silently stalling — worth catching before a session times out
         with nothing checkpointed.
    """

    def __init__(self, max_save_gap_seconds: int = 600):
        self.max_save_gap_seconds = max_save_gap_seconds
        self._last_save_time: Optional[float] = None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        now = time.time()
        if self._last_save_time is not None:
            gap = now - self._last_save_time
            if gap > self.max_save_gap_seconds:
                print(
                    f"[DriveCheckpointGuard] Warning: {gap:.0f}s since last "
                    f"checkpoint (threshold {self.max_save_gap_seconds}s). "
                    f"Check for a data-loading stall."
                )
        self._last_save_time = now

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        state_record: Dict = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "log_history_tail": state.log_history[-5:] if state.log_history else [],
            "timestamp": now,
        }
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "run_state.json"), "w") as f:
                json.dump(state_record, f, indent=2)
        except OSError as e:
            print(f"[DriveCheckpointGuard] Could not write run_state.json: {e}")

        return control


class LossSpikeMonitor(TrainerCallback):
    """
    Flags sudden loss spikes (>2x the trailing average) which, on a small
    1.5B model with 4-bit quantization, are usually a sign of an unstable
    learning rate or a bad batch (e.g. truncated/garbled text) rather than
    something to just let ride out.
    """

    def __init__(self, window: int = 20, spike_ratio: float = 2.0):
        self.window = window
        self.spike_ratio = spike_ratio
        self._recent_losses = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ):
        if not logs or "loss" not in logs:
            return control

        loss = logs["loss"]
        if self._recent_losses:
            avg = sum(self._recent_losses) / len(self._recent_losses)
            if avg > 0 and loss > avg * self.spike_ratio:
                print(
                    f"[LossSpikeMonitor] step {state.global_step}: loss {loss:.4f} "
                    f"is >{self.spike_ratio}x the trailing avg ({avg:.4f})"
                )

        self._recent_losses.append(loss)
        if len(self._recent_losses) > self.window:
            self._recent_losses.pop(0)

        return control
