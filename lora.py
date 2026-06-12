"""
lora.py — LoRA/QLoRA configuration for Gemma 4-4B
CRITICAL FIX: Add dummy prepare_inputs_for_generation to bypass PEFT crash
"""

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import (
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)


# ============================================================
# GEMMA 4-4B LORA CONFIG
# ============================================================

def create_lora_config(
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        use_rslora: bool = False,
) -> LoraConfig:
    """Create LoRA config for Gemma 4-4B."""

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
        modules_to_save=None,
    )

    return config


# ============================================================
# QLoRA 4-bit CONFIG
# ============================================================

def create_bnb_config(
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
) -> BitsAndBytesConfig:
    """Create BitsAndBytes config for QLoRA."""

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )
    else:
        return BitsAndBytesConfig(load_in_8bit=True)


# ============================================================
# MODEL WRAPPER — BULLETPROOF FOR GEMMA 4
# ============================================================

class Gemma4MultimodalTrainer(nn.Module):
    """
    Wrapper for Gemma 4-4B with LoRA.

    CRITICAL FIX: Gemma4TextModel lacks prepare_inputs_for_generation,
    which PEFT's get_peft_model requires. We add a dummy method before
    applying LoRA, then remove it afterward.
    """

    def __init__(
            self,
            model_name: str = "google/gemma-4-e4b-it",
            lora_config: LoraConfig = None,
            bnb_config: BitsAndBytesConfig = None,
            device_map: str = "auto",
            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.model_name = model_name

        print(f"🔄 Loading Gemma 4-4B from {model_name}...")

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config

        # Load the full multimodal model
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Access the inner Gemma4Model
        inner_model = self.model.model  # Gemma4Model

        # Get config details from text config
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size
        self.num_layers = text_config.num_hidden_layers

        print(f"   Text hidden size: {self.hidden_size}")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Layers: {self.num_layers}")

        # Print encoder info (with safe attribute access)
        try:
            vision_params = sum(p.numel() for p in inner_model.vision_model.parameters())
            print(f"   Vision encoder: ~{vision_params / 1e6:.0f}M params")
        except AttributeError:
            for attr_name in ['vision_tower', 'vision_encoder', 'visual_model']:
                if hasattr(inner_model, attr_name):
                    vision_params = sum(p.numel() for p in getattr(inner_model, attr_name).parameters())
                    print(f"   Vision encoder (via {attr_name}): ~{vision_params / 1e6:.0f}M params")
                    break

        try:
            audio_params = sum(p.numel() for p in inner_model.audio_model.parameters())
            print(f"   Audio encoder: ~{audio_params / 1e6:.0f}M params")
        except AttributeError:
            print("   Audio encoder: attribute not found")

        # Apply LoRA to language model
        if lora_config:
            print(f"🎯 Applying LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha})...")

            lm = inner_model.language_model

            # CRITICAL FIX: Add dummy prepare_inputs_for_generation
            # PEFT's get_peft_model requires this attribute on the base model
            if not hasattr(lm, 'prepare_inputs_for_generation'):
                print("   📝 Adding dummy prepare_inputs_for_generation for PEFT compatibility...")

                def dummy_prepare_inputs(*args, **kwargs):
                    """Dummy method to satisfy PEFT's requirements."""
                    return kwargs

                lm.prepare_inputs_for_generation = dummy_prepare_inputs

            # For 4-bit models, also need to handle gradient requirements
            if bnb_config and bnb_config.load_in_4bit:
                print("   📝 Enabling input gradients for 4-bit training...")

                # Enable gradient checkpointing
                if hasattr(lm, 'gradient_checkpointing_enable'):
                    lm.gradient_checkpointing_enable()
                    print("   ✅ Gradient checkpointing enabled")

                # Enable input requires grads for all parameters
                for param in lm.parameters():
                    if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        param.requires_grad = True

            # Apply LoRA using get_peft_model
            inner_model.language_model = get_peft_model(lm, lora_config)
            inner_model.language_model.print_trainable_parameters()

        # Freeze vision and audio encoders
        try:
            for param in inner_model.vision_model.parameters():
                param.requires_grad = False
            print("   ✅ Vision encoder frozen")
        except AttributeError:
            print("   ⚠️ Could not freeze vision encoder")

        try:
            for param in inner_model.audio_model.parameters():
                param.requires_grad = False
            print("   ✅ Audio encoder frozen")
        except AttributeError:
            print("   ⚠️ Could not freeze audio encoder")

        print("✅ Model initialized!")

    def forward(self, **kwargs):
        """Forward pass — delegate to underlying model."""
        return self.model(**kwargs)

    def generate(self, **kwargs):
        """Generation — delegate to underlying model."""
        return self.model.generate(**kwargs)

    def save_pretrained(self, save_path: str):
        """Save model (LoRA adapters + full model)."""
        import os
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"💾 Model saved to {save_path}")

    def get_trainable_params(self):
        """Get number of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ============================================================
# FACTORY
# ============================================================

def create_model(
        model_name: str = "google/gemma-4-e4b-it",
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device_map: str = "auto",
) -> Gemma4MultimodalTrainer:
    """Factory to create model with LoRA/QLoRA."""

    lora_config = create_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    bnb_config = None
    if use_qlora:
        bnb_config = create_bnb_config(load_in_4bit=True)

    model = Gemma4MultimodalTrainer(
        model_name=model_name,
        lora_config=lora_config,
        bnb_config=bnb_config,
        device_map=device_map,
    )

    return model