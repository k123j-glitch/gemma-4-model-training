"""
utils.py — Shared utilities for Gemma 4-4B multimodal training
Handles token constants, checkpointing, memory management
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# ============================================================
# GEMMA 4 SPECIAL TOKEN IDs (from official HF config)
# ============================================================

IMAGE_TOKEN_ID = 258880  # <|image|> placeholder
AUDIO_TOKEN_ID = 258881  # <|audio|> placeholder
BOI_TOKEN_ID = 255999  # <|boi|> begin-of-image
EOI_TOKEN_ID = 258882  # <|eoi|> end-of-image
BOA_TOKEN_ID = 256000  # <|boa|> begin-of-audio
EOA_TOKEN_ID = 258883  # <|eoa|> end-of-audio
VIDEO_TOKEN_ID = 258884  # <|video|> placeholder

# Default soft token counts
DEFAULT_IMAGE_SOFT_TOKENS = 280  # configurable: 70, 140, 280, 560, 1120
DEFAULT_AUDIO_SOFT_TOKENS = 750  # max cap, dynamic based on duration
AUDIO_MS_PER_TOKEN = 40  # 40ms per audio soft token

# Modality type IDs for mm_token_type_ids
MODALITY_TEXT = 0
MODALITY_IMAGE = 1
MODALITY_VIDEO = 2


# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_modality_from_sample(sample: Dict) -> str:
    """Determine modality from sample."""
    has_image = bool(sample.get("image_path"))
    has_audio = bool(sample.get("audio_path"))
    if has_image and has_audio:
        return "multimodal"
    elif has_image:
        return "vision"
    elif has_audio:
        return "audio"
    return "text"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================
# MEMORY UTILS
# ============================================================

def get_gpu_memory():
    """Get GPU memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU: {allocated:.2f}GB / {total:.2f}GB (reserved: {reserved:.2f}GB)"
    return "No CUDA"


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        loss: float,
        path: str,
        is_best: bool = False,
):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    torch.save(checkpoint, path)
    if is_best:
        best_path = str(Path(path).with_suffix(".best.pt"))
        torch.save(checkpoint, best_path)

    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path: str):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"📂 Checkpoint loaded: {path}")
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


# ============================================================
# LABEL MASKING FOR CHAT FORMAT
# ============================================================

def create_labels_for_chat(
        input_ids: List[int],
        tokenizer,
        user_token_id: Optional[int] = None,
        model_token_id: Optional[int] = None,
) -> List[int]:
    """
    Create labels with loss masking.
    Only compute loss on assistant (model) response tokens.

    Gemma 4 chat format:
    <start_of_turn>user\n...<end_of_turn>
    <start_of_turn>model\n...<end_of_turn>
    """
    labels = [-100] * len(input_ids)

    # Decode to find assistant sections
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Find all model/assistant response sections
    import re
    # Pattern matches content between <start_of_turn>model\n and <end_of_turn>
    pattern = r'<start_of_turn>model\n(.*?)<end_of_turn>'
    matches = list(re.finditer(pattern, text, re.DOTALL))

    for match in matches:
        start_char = match.start(1)
        end_char = match.end(1)

        # Convert char positions to token positions
        prefix = text[:start_char]
        response = text[start_char:end_char]

        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]

        start_token = len(prefix_tokens)
        end_token = start_token + len(response_tokens)

        for i in range(start_token, min(end_token, len(labels))):
            if i < len(labels):
                labels[i] = input_ids[i]

    return labels


# ============================================================
# AUDIO DURATION HELPERS
# ============================================================

def get_audio_duration_ms(audio_path: str) -> int:
    """Get audio duration in milliseconds."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return int(info.duration * 1000)
    except Exception:
        # Fallback: assume 30s max
        return 30000


def compute_audio_token_count(duration_ms: int, ms_per_token: int = AUDIO_MS_PER_TOKEN) -> int:
    """Compute number of audio soft tokens for given duration."""
    import math
    return min(math.ceil(duration_ms / ms_per_token), DEFAULT_AUDIO_SOFT_TOKENS)