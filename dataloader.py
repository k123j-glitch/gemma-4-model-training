"""
dataloader.py — Multimodal DataLoader for Gemma 4-4B
Uses official Gemma4Processor for proper early fusion preprocessing
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from utils import (
    load_jsonl, get_modality_from_sample,
    DEFAULT_IMAGE_SOFT_TOKENS, DEFAULT_AUDIO_SOFT_TOKENS,
)


# ============================================================
# MULTIMODAL DATASET
# ============================================================

class MultimodalDataset(Dataset):
    """
    Dataset for Gemma 4-4B multimodal training.

    Uses Gemma4Processor internally for consistent preprocessing.
    Stores raw paths; processor is applied in collate_fn for batching.
    """

    def __init__(
            self,
            data_path: str,
            processor: AutoProcessor,
            max_length: int = 4096,
    ):
        self.data = load_jsonl(data_path)
        self.processor = processor
        self.max_length = max_length

        print(f"📊 Loaded {len(self.data)} samples from {data_path}")

        # Print modality distribution
        modalities = {}
        for item in self.data:
            mod = get_modality_from_sample(item)
            modalities[mod] = modalities.get(mod, 0) + 1
        print(f"   Modality distribution: {modalities}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        modality = get_modality_from_sample(sample)

        # Extract messages
        messages = sample.get("messages", [])

        # Build content list for processor
        # IMPORTANT: Gemma 4 expects image BEFORE text, audio AFTER text
        content = []

        # Add image if present (goes BEFORE text)
        image_path = sample.get("image_path")
        image_obj = None
        if image_path:
            full_path = os.path.join("data", image_path)
            if os.path.exists(full_path):
                try:
                    image_obj = Image.open(full_path).convert("RGB")
                    content.append({"type": "image"})
                except Exception as e:
                    print(f"Warning: Failed to load image {full_path}: {e}")

        # Add text from messages
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            # Replace <image> and <audio> placeholders
            text = text.replace("<image>", "").replace("<audio>", "").strip()
            if text:
                text_parts.append(text)

        if text_parts:
            content.append({"type": "text", "text": " ".join(text_parts)})

        # Add audio if present (goes AFTER text)
        audio_path = sample.get("audio_path")
        audio_array = None
        if audio_path:
            full_path = os.path.join("data", audio_path)
            if os.path.exists(full_path):
                try:
                    import soundfile as sf
                    audio_array, sr = sf.read(full_path)
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)
                    content.append({"type": "audio"})
                except Exception as e:
                    print(f"Warning: Failed to load audio {full_path}: {e}")

        # Build message for processor
        processor_message = {
            "role": "user",
            "content": content
        }

        result = {
            "messages": [processor_message],
            "modality": modality,
            "source": sample.get("source", "unknown"),
            "image": image_obj,  # PIL Image or None
            "audio": audio_array,  # numpy array or None
        }

        return result


# ============================================================
# COLLATE FUNCTION — USES GEMMA4 PROCESSOR
# ============================================================

def multimodal_collate_fn(batch: List[Dict], processor: AutoProcessor, max_length: int = 4096) -> Dict[str, Any]:
    """
    Collate function that uses Gemma4Processor for proper preprocessing.

    CRITICAL: Returns ONLY tensors that the model expects.
    Metadata (modality, source) is stored separately and must NOT be passed to model.forward().
    """

    # Separate by modality for efficient batching
    texts = []
    images = []
    audios = []

    for item in batch:
        # Apply chat template to get text
        text = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

        images.append(item["image"])  # None or PIL Image
        audios.append(item["audio"])  # None or numpy array

    # Prepare processor inputs
    # Filter out None images/audios for processor
    valid_images = [img for img in images if img is not None]
    valid_audios = [audio for audio in audios if audio is not None]

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }

    if valid_images:
        processor_kwargs["images"] = images  # Pass full list with Nones

    if valid_audios:
        processor_kwargs["audio"] = valid_audios  # Only valid audio arrays

    # Process through Gemma4Processor
    inputs = processor(**processor_kwargs)

    # Create labels: mask padding tokens and special modality tokens
    labels = inputs["input_ids"].clone()

    # Mask padding tokens
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100

    # Mask image/audio special tokens (don't compute loss on them)
    # These are placeholder tokens that get replaced by encoder outputs
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        for special_token in ['image_token_id', 'boi_token_id', 'eoi_token_id',
                              'audio_token_id', 'boa_token_id', 'eoa_token_id']:
            if hasattr(tokenizer, special_token):
                token_id = getattr(tokenizer, special_token)
                labels[labels == token_id] = -100

    inputs["labels"] = labels

    # CRITICAL FIX: Store metadata separately, don't include in returned dict
    # The model's forward() will receive **inputs, and 'source'/'modality' 
    # conflict with loss function parameters
    # We return a dict with ONLY model-expected keys + metadata in a sub-dict
    result = dict(inputs)  # Copy tensor inputs
    result["_metadata"] = {
        "modality": [item["modality"] for item in batch],
        "source": [item["source"] for item in batch],
    }

    return result


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloaders(
        train_path: str,
        eval_path: str,
        model_name: str = "google/gemma-4-e4b-it",
        batch_size: int = 1,
        num_workers: int = 0,  # Must be 0 for processor in collate
        max_length: int = 4096,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders."""

    print(f"📝 Loading Gemma4Processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # Required for Gemma 4 generation
    )

    train_dataset = MultimodalDataset(
        data_path=train_path,
        processor=processor,
        max_length=max_length,
    )

    eval_dataset = MultimodalDataset(
        data_path=eval_path,
        processor=processor,
        max_length=max_length,
    )

    # Create partial collate function with processor bound
    from functools import partial
    collate_fn = partial(multimodal_collate_fn, processor=processor, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, eval_loader