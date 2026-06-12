"""
train.py — Training script for Gemma 4-4B Multimodal
Uses official AutoModelForMultimodalLM with LoRA/QLoRA
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from utils import (
    save_checkpoint, load_checkpoint, get_gpu_memory,
    cleanup_memory, count_parameters,
)
from lora import create_model
from dataloader import create_dataloaders


# ============================================================
# CONFIG
# ============================================================

class TrainingConfig:
    """Training hyperparameters for Gemma 4-4B."""

    model_name: str = "google/gemma-4-e4b-it"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # QLoRA
    use_qlora: bool = True

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Precision
    mixed_precision: str = "bf16"

    # Checkpointing
    save_every: int = 500
    eval_every: int = 250
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Data
    train_path: str = "data/ready/train.jsonl"
    eval_path: str = "data/ready/eval.jsonl"

    # System
    num_workers: int = 0  # Must be 0 for processor in collate_fn
    seed: int = 42


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        run_name = f"gemma4-e4b-lora{config.lora_r}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(f"{config.log_dir}/{run_name}")

        print(f"🚀 Initializing Trainer: {run_name}")
        print(f"   Device: {self.device}")
        print(f"   Mixed precision: {config.mixed_precision}")

        # Create model
        print("🤖 Creating model...")
        self.model = create_model(
            model_name=config.model_name,
            use_qlora=config.use_qlora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        # Count parameters
        trainable, total = self.model.get_trainable_params()
        print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Create dataloaders
        print("📊 Loading datasets...")
        self.train_loader, self.eval_loader = create_dataloaders(
            train_path=config.train_path,
            eval_path=config.eval_path,
            model_name=config.model_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_length=config.max_length,
        )

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("🔥 STARTING TRAINING")
        print("=" * 60)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            save_path = f"{self.config.output_dir}/epoch_{epoch}.pt"
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, 0.0, save_path,
            )

        # Final save
        final_path = f"{self.config.output_dir}/final"
        self.model.save_pretrained(final_path)
        print(f"\n🎉 Training complete! Model saved to {final_path}")

    def _train_epoch(self):
        """Train one epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # CRITICAL FIX: Remove metadata before passing to model
            # Metadata keys like 'source' conflict with loss function parameters
            metadata = batch.pop("_metadata", None)

            # Move tensors to device
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device)
                else:
                    device_batch[k] = v

            # Forward with mixed precision
            autocast_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=self.config.mixed_precision != "no"):
                outputs = self.model(**device_batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                current_loss = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "step": self.global_step,
                })

                if self.global_step % 10 == 0:
                    self.writer.add_scalar("train/loss", current_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

                if self.global_step % self.config.eval_every == 0:
                    eval_loss = self._evaluate()
                    self.model.train()

                    self.writer.add_scalar("eval/loss", eval_loss, self.global_step)

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        save_path = f"{self.config.output_dir}/best.pt"
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.epoch, self.global_step, eval_loss,
                            save_path, is_best=True,
                        )

                if self.global_step % self.config.save_every == 0:
                    save_path = f"{self.config.output_dir}/step_{self.global_step}.pt"
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.epoch, self.global_step, current_loss,
                        save_path,
                    )

                if self.global_step % 100 == 0:
                    print(f"\n   {get_gpu_memory()}")
                    cleanup_memory()

        avg_loss = epoch_loss / num_batches
        print(f"\n✅ Epoch {self.epoch} complete. Avg loss: {avg_loss:.4f}")

    @torch.no_grad()
    def _evaluate(self):
        """Run evaluation."""
        print(f"\n🔍 Evaluating at step {self.global_step}...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_loader, desc="Eval", ncols=60):
            # CRITICAL FIX: Remove metadata before passing to model
            metadata = batch.pop("_metadata", None)

            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device)
                else:
                    device_batch[k] = v

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(**device_batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"   Eval loss: {avg_loss:.4f}")
        return avg_loss


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Gemma 4-4B Multimodal")
    parser.add_argument("--model", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--train", type=str, default="data/ready/train.jsonl")
    parser.add_argument("--eval", type=str, default="data/ready/eval.jsonl")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig()
    config.model_name = args.model
    config.train_path = args.train
    config.eval_path = args.eval
    config.output_dir = args.output
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.learning_rate = args.lr
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.max_length = args.max_length
    config.use_qlora = not args.no_qlora

    if not Path(config.train_path).exists():
        print(f"❌ Train data not found: {config.train_path}")
        sys.exit(1)

    trainer = Trainer(config)

    if args.resume:
        print(f"📂 Resuming from {args.resume}")
        epoch, step, loss = load_checkpoint(
            trainer.model, trainer.optimizer, trainer.scheduler, args.resume
        )
        trainer.epoch = epoch
        trainer.global_step = step

    trainer.train()


if __name__ == "__main__":
    main()