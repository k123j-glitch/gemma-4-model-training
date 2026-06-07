"""
Multimodal Dataset Downloader for Gemma 4-4B (E4B)
Saves BOTH .jsonl (training) and .json (human-readable) formats
Fixed OpenHermes-2.5 parsing (ShareGPT format)
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import requests

# ============================================================
# CONFIGURATION
# ============================================================

MAX_TEXT_SAMPLES = 50_000
MAX_VISION_SAMPLES = 10_000
MAX_AUDIO_SAMPLES = 5_000
MAX_MM_SAMPLES = 5_000

OUTPUT_DIR = "data"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# HELPER: SAVE BOTH JSONL AND JSON
# ============================================================

def save_both_formats(data: List[Dict], filepath: Path):
    """
    Save dataset in BOTH formats:
    - .jsonl  -> for training (line-delimited, efficient)
    - .json   -> for human inspection (pretty-printed array)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save as JSONL (for training pipelines)
    jsonl_path = filepath.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 2. Save as JSON (for human viewing)
    json_path = filepath.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"     💾 Saved: {jsonl_path.name} ({len(data):,} items)")
    print(f"     💾 Saved: {json_path.name}  (human-readable)")


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file into a list."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def preview_data(data: List[Dict], n: int = 3, modality: str = "text"):
    """Print a nice preview of the first n examples."""
    print(f"\n{'=' * 60}")
    print(f"👁️  PREVIEW: First {n} examples ({modality})")
    print(f"{'=' * 60}")

    for i, item in enumerate(data[:n]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Source: {item.get('source', 'unknown')}")
        print(f"Modality: {item.get('modality', 'unknown')}")

        if "messages" in item:
            for msg in item["messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content for preview
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"  [{role.upper()}]: {content}")
        elif "instruction" in item and "response" in item:
            print(f"  [USER]: {item['instruction'][:300]}")
            print(f"  [ASSISTANT]: {item['response'][:300]}")

        if "image_path" in item and item["image_path"]:
            print(f"  [IMAGE]: {item['image_path']}")
        if "audio_path" in item and item["audio_path"]:
            print(f"  [AUDIO]: {item['audio_path']}")


# ============================================================
# TEXT DATASETS (FIXED OpenHermes parsing!)
# ============================================================

def download_text_datasets():
    """Download text instruction datasets."""

    print("\n" + "=" * 60)
    print("📝 TEXT INSTRUCTION DATASETS")
    print("=" * 60)

    text_configs = [
        {
            "name": "OpenHermes-2.5",
            "repo": "teknium/OpenHermes-2.5",
            "split": "train",
            "format": "sharegpt",  # FIXED: uses conversations with from/value
            "max_samples": MAX_TEXT_SAMPLES
        },
        {
            "name": "UltraChat-200k",
            "repo": "HuggingFaceH4/ultrachat_200k",
            "split": "train_sft",
            "format": "messages",  # messages with role/content
            "max_samples": MAX_TEXT_SAMPLES
        },
        {
            "name": "FineTome-100k",
            "repo": "mlabonne/FineTome-100k",
            "split": "train",
            "format": "conversations",  # conversations array
            "max_samples": MAX_TEXT_SAMPLES
        },
        {
            "name": "OpenOrca",
            "repo": "Open-Orca/OpenOrca",
            "split": "train",
            "format": "question_answer",  # question, response
            "max_samples": MAX_TEXT_SAMPLES
        }
    ]

    all_text = []

    for cfg in text_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True, trust_remote_code=True)
            examples = []

            for i, item in enumerate(tqdm(ds, total=cfg["max_samples"], desc=f"  {cfg['name']}", ncols=60)):
                if i >= cfg["max_samples"]:
                    break

                parsed = parse_text_item(item, cfg["format"], cfg["name"])
                if parsed:
                    examples.append(parsed)

            # Save BOTH formats
            save_path = Path(OUTPUT_DIR) / "text" / cfg["name"]
            save_both_formats(examples, save_path)

            # Preview first 2 examples
            preview_data(examples, n=2, modality="text")

            all_text.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_text


def parse_text_item(item: Dict, fmt: str, source: str) -> Optional[Dict]:
    """Parse a text dataset item into unified Gemma 4 chat format."""

    # FIXED: OpenHermes uses ShareGPT format!
    # {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    if fmt == "sharegpt":
        conv = item.get("conversations", [])
        if len(conv) >= 2:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = []
            for c in conv:
                role = role_map.get(c.get("from", "user"), "user")
                content = c.get("value", "")
                if content:  # Skip empty messages
                    messages.append({"role": role, "content": content})
            if len(messages) >= 2:
                return {
                    "modality": "text",
                    "source": source,
                    "messages": messages
                }

    # Format: {"messages": [{"role": "user", "content": "..."}, ...]}
    elif fmt == "messages":
        msgs = item.get("messages", [])
        if len(msgs) >= 2:
            return {
                "modality": "text",
                "source": source,
                "messages": [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in msgs[:10]
                ]
            }

    # Format: {"conversations": [{"from": "human", "value": "..."}, ...]}
    elif fmt == "conversations":
        conv = item.get("conversations", [])
        if len(conv) >= 2:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            return {
                "modality": "text",
                "source": source,
                "messages": [
                    {"role": role_map.get(c.get("from", "user"), "user"), "content": c.get("value", "")}
                    for c in conv[:10]
                ]
            }

    # Format: {"instruction": "...", "output": "..."}
    elif fmt == "instruction_output":
        instruction = item.get("instruction") or item.get("input", "")
        response = item.get("output") or item.get("response", "")
        if instruction and response:
            return {
                "modality": "text",
                "source": source,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }

    # Format: {"question": "...", "response": "..."}
    elif fmt == "question_answer":
        question = item.get("question") or item.get("query", "")
        response = item.get("response") or item.get("answer", "")
        if question and response:
            return {
                "modality": "text",
                "source": source,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
            }

    return None


# ============================================================
# VISION-LANGUAGE DATASETS
# ============================================================

def download_vision_datasets():
    """Download vision-language datasets with actual image downloads."""

    print("\n" + "=" * 60)
    print("🖼️ VISION-LANGUAGE DATASETS")
    print("=" * 60)

    vision_configs = [
        {
            "name": "LLaVA-Instruct-150K",
            "repo": "liuhaotian/LLaVA-Instruct-150K",
            "split": "train",
            "image_key": "image",
            "text_key": "conversations",
            "max_samples": MAX_VISION_SAMPLES
        },
        {
            "name": "COCO-Captions",
            "repo": "yerevann/coco-karpathy",
            "split": "train",
            "image_key": "image",
            "text_key": "sentences",
            "max_samples": MAX_VISION_SAMPLES
        }
    ]

    all_vision = []
    image_dir = Path(OUTPUT_DIR) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for cfg in vision_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True, trust_remote_code=True)
            examples = []
            img_counter = 0

            for i, item in enumerate(tqdm(ds, total=cfg["max_samples"], desc=f"  {cfg['name']}", ncols=60)):
                if i >= cfg["max_samples"]:
                    break

                parsed = parse_vision_item(item, cfg, image_dir, img_counter)
                if parsed:
                    examples.append(parsed)
                    img_counter += 1

            # Save BOTH formats
            save_path = Path(OUTPUT_DIR) / "vision" / cfg["name"]
            save_both_formats(examples, save_path)

            preview_data(examples, n=2, modality="vision")

            all_vision.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples, {img_counter} images")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_vision


def parse_vision_item(item: Dict, cfg: Dict, img_folder: Path, img_idx: int) -> Optional[Dict]:
    """Parse a vision dataset item and save the image."""

    img_folder.mkdir(parents=True, exist_ok=True)

    # Extract image
    image_path = None
    if "image" in item and item["image"] is not None:
        try:
            pil_image = item["image"]
            img_filename = f"{cfg['name']}_{img_idx:06d}.jpg"
            img_path = img_folder / img_filename
            pil_image.save(img_path)
            image_path = str(img_path.relative_to(OUTPUT_DIR))
        except Exception:
            pass

    # Extract text
    text = ""
    response = ""
    if "conversations" in item:
        conv = item["conversations"]
        if isinstance(conv, list) and len(conv) >= 2:
            text = conv[0].get("value", "") if isinstance(conv[0], dict) else str(conv[0])
            response = conv[1].get("value", "") if isinstance(conv[1], dict) else str(conv[1])
    elif "caption" in item:
        text = item["caption"]
        response = item["caption"]
    elif "sentences" in item:
        sents = item["sentences"]
        text = sents[0] if isinstance(sents, list) else str(sents)
        response = text

    if text and image_path:
        return {
            "modality": "vision",
            "source": cfg["name"],
            "image_path": image_path,
            "messages": [
                {"role": "user", "content": f"<image>\n{text}"},
                {"role": "assistant", "content": response}
            ]
        }

    return None


# ============================================================
# AUDIO DATASETS
# ============================================================

def download_audio_datasets():
    """Download audio datasets."""

    print("\n" + "=" * 60)
    print("🎵 AUDIO DATASETS")
    print("=" * 60)

    audio_configs = [
        {
            "name": "CommonVoice-17",
            "repo": "mozilla-foundation/common_voice_17_0",
            "config": "en",
            "split": "train",
            "max_samples": MAX_AUDIO_SAMPLES,
            "text_key": "sentence"
        },
        {
            "name": "GigaSpeech",
            "repo": "speechcolab/gigaspeech",
            "config": "xl",
            "split": "train",
            "max_samples": MAX_AUDIO_SAMPLES,
            "text_key": "text"
        }
    ]

    all_audio = []
    audio_dir = Path(OUTPUT_DIR) / "audio_files"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for cfg in audio_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            load_kwargs = {"split": cfg["split"], "streaming": True, "trust_remote_code": True}
            if cfg.get("config"):
                load_kwargs["name"] = cfg["config"]

            ds = load_dataset(cfg["repo"], **load_kwargs)
            examples = []
            audio_counter = 0

            for i, item in enumerate(tqdm(ds, total=cfg["max_samples"], desc=f"  {cfg['name']}", ncols=60)):
                if i >= cfg["max_samples"]:
                    break

                parsed = parse_audio_item(item, cfg, audio_dir, audio_counter)
                if parsed:
                    examples.append(parsed)
                    audio_counter += 1

            # Save BOTH formats
            save_path = Path(OUTPUT_DIR) / "audio" / cfg["name"]
            save_both_formats(examples, save_path)

            preview_data(examples, n=2, modality="audio")

            all_audio.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples, {audio_counter} audio files")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_audio


def parse_audio_item(item: Dict, cfg: Dict, audio_folder: Path, audio_idx: int) -> Optional[Dict]:
    """Parse an audio dataset item and save the audio."""

    audio_folder.mkdir(parents=True, exist_ok=True)

    # Extract text
    text = item.get(cfg["text_key"], "")
    if not text:
        return None

    # Try to save audio
    audio_path = None
    if "audio" in item and item["audio"] is not None:
        try:
            audio_data = item["audio"]
            if isinstance(audio_data, dict):
                orig_path = audio_data.get("path", "")
                ext = Path(orig_path).suffix if orig_path else ".wav"
                if not ext or ext == ".":
                    ext = ".wav"

                audio_filename = f"{cfg['name']}_{audio_idx:06d}{ext}"
                audio_save_path = audio_folder / audio_filename

                if "array" in audio_data and "sampling_rate" in audio_data:
                    try:
                        import soundfile as sf
                        sf.write(audio_save_path, audio_data["array"], audio_data["sampling_rate"])
                        audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                    except ImportError:
                        # Fallback: just note the path
                        audio_path = orig_path
        except Exception:
            pass

    if audio_path is None:
        audio_path = item.get("path", "") or ""

    prompt = "Transcribe this audio" if "voice" in cfg["name"].lower() or "common" in cfg[
        "name"].lower() else "Describe this audio"

    return {
        "modality": "audio",
        "source": cfg["name"],
        "audio_path": audio_path,
        "messages": [
            {"role": "user", "content": f"<audio>\n{prompt}"},
            {"role": "assistant", "content": text}
        ]
    }


# ============================================================
# COMBINE & SPLIT (BOTH FORMATS)
# ============================================================

def combine_and_split(all_data: List[Dict], train_ratio: float = 0.92):
    """Shuffle, split into train/eval, and save in BOTH formats."""

    print("\n" + "=" * 60)
    print("🔗 COMBINING & SPLITTING DATASETS")
    print("=" * 60)

    random.shuffle(all_data)

    split_idx = int(train_ratio * len(all_data))
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    ready_dir = Path(OUTPUT_DIR) / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    # Save train in BOTH formats
    train_jsonl = ready_dir / "train.jsonl"
    train_json = ready_dir / "train.json"
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Save eval in BOTH formats
    eval_jsonl = ready_dir / "eval.jsonl"
    eval_json = ready_dir / "eval.json"
    with open(eval_jsonl, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    # Stats
    stats = {
        "total_samples": len(all_data),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "modalities": {}
    }
    for modality in ["text", "vision", "audio", "multimodal"]:
        count = sum(1 for d in all_data if d.get("modality") == modality)
        stats["modalities"][modality] = count

    stats_path = ready_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  ✅ Train: {len(train_data):,} examples")
    print(f"     📄 {train_jsonl.name}  (for training)")
    print(f"     📄 {train_json.name}   (for viewing)")
    print(f"\n  ✅ Eval:  {len(eval_data):,} examples")
    print(f"     📄 {eval_jsonl.name}  (for training)")
    print(f"     📄 {eval_json.name}   (for viewing)")
    print(f"\n  📊 Stats: {stats_path}")

    return train_data, eval_data


def print_summary(all_data: List[Dict]):
    """Print dataset summary."""

    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)

    sources = {}
    for item in all_data:
        src = item.get("source", "unknown")
        mod = item.get("modality", "unknown")
        key = f"{mod}/{src}"
        sources[key] = sources.get(key, 0) + 1

    for key, count in sorted(sources.items()):
        print(f"  {key:<40} {count:>8,}")

    print(f"\n  {'TOTAL':<40} {len(all_data):>8,}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Gemma 4-4B Multimodal Dataset Downloader")
    print("   Outputs: .jsonl (training) + .json (human-readable)")
    print("   FIXED: OpenHermes-2.5 ShareGPT format parsing!")
    print("=" * 60)

    # Clear previous data if needed
    if Path(OUTPUT_DIR).exists():
        response = input(f"\n⚠️  '{OUTPUT_DIR}/' already exists. Delete? [y/N]: ").strip().lower()
        if response == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print("   Cleared existing data.")

    # Download all modalities
    text_data = download_text_datasets()
    vision_data = download_vision_datasets()
    audio_data = download_audio_datasets()

    # Combine
    all_data = text_data + vision_data + audio_data

    if not all_data:
        print("\n❌ No data downloaded. Check your internet connection.")
        return

    print_summary(all_data)
    combine_and_split(all_data)

    print("\n" + "=" * 60)
    print("🎉 DONE! Your datasets are ready:")
    print("   📁 data/ready/train.json  <- Open this to inspect data!")
    print("   📁 data/ready/train.jsonl <- Use this for training")
    print("   📁 data/ready/eval.json   <- Open this to inspect data!")
    print("   📁 data/ready/eval.jsonl  <- Use this for training")
    print("=" * 60)


if __name__ == "__main__":
    main()