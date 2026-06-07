"""
Multimodal Dataset Downloader for Gemma 4-4B (E4B)
FIXED: Actually downloads images and audio files
"""

import os
import json
import random
import shutil
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import requests

# ============================================================
# CONFIGURATION
# ============================================================

MAX_TEXT_SAMPLES = 10_000      # Reduced for memory
MAX_VISION_SAMPLES = 5_000
MAX_AUDIO_SAMPLES = 2_000
MAX_MM_SAMPLES = 5_000

OUTPUT_DIR = "data"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def save_both_formats(data: List[Dict], filepath: Path):
    """Save dataset in BOTH .jsonl and .json formats."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = filepath.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    json_path = filepath.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"     💾 Saved: {jsonl_path.name} ({len(data):,} items)")
    print(f"     💾 Saved: {json_path.name}  (human-readable)")


def preview_data(data: List[Dict], n: int = 2, modality: str = "text"):
    """Print preview of first n examples."""
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
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"  [{role.upper()}]: {content}")

        if "image_path" in item and item["image_path"]:
            print(f"  [IMAGE]: {item['image_path']}")
        if "audio_path" in item and item["audio_path"]:
            print(f"  [AUDIO]: {item['audio_path']}")


# ============================================================
# TEXT DATASETS
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
            "format": "sharegpt",
            "max_samples": MAX_TEXT_SAMPLES
        },
        {
            "name": "UltraChat-200k",
            "repo": "HuggingFaceH4/ultrachat_200k",
            "split": "train_sft",
            "format": "messages",
            "max_samples": MAX_TEXT_SAMPLES
        },
        {
            "name": "FineTome-100k",
            "repo": "mlabonne/FineTome-100k",
            "split": "train",
            "format": "instruction_output",
            "max_samples": MAX_TEXT_SAMPLES
        },
    ]

    all_text = []

    for cfg in text_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            # Use streaming for text (no binary data needed)
            ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True, trust_remote_code=True)
            examples = []

            for i, item in enumerate(tqdm(ds, total=cfg["max_samples"], desc=f"  {cfg['name']}", ncols=60)):
                if i >= cfg["max_samples"]:
                    break

                parsed = parse_text_item(item, cfg["format"], cfg["name"])
                if parsed:
                    examples.append(parsed)

            save_path = Path(OUTPUT_DIR) / "text" / cfg["name"]
            save_both_formats(examples, save_path)
            preview_data(examples, n=2, modality="text")

            all_text.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_text


def parse_text_item(item: Dict, fmt: str, source: str) -> Optional[Dict]:
    """Parse text dataset item into unified chat format."""

    if fmt == "sharegpt":
        conv = item.get("conversations", [])
        if len(conv) >= 2:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = []
            for c in conv:
                role = role_map.get(c.get("from", "user"), "user")
                content = c.get("value", "")
                if content:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 2:
                return {"modality": "text", "source": source, "messages": messages}

    elif fmt == "messages":
        msgs = item.get("messages", [])
        if len(msgs) >= 2:
            valid_msgs = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in msgs[:10] if m.get("content", "")
            ]
            if len(valid_msgs) >= 2:
                return {"modality": "text", "source": source, "messages": valid_msgs}

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

    return None


# ============================================================
# VISION DATASETS — FIXED: Actually downloads images
# ============================================================

def download_vision_datasets():
    """Download vision-language datasets with ACTUAL image downloads."""

    print("\n" + "=" * 60)
    print("🖼️ VISION-LANGUAGE DATASETS")
    print("=" * 60)

    vision_configs = [
        {
            "name": "COCO-Captions",
            "repo": "yerevann/coco-karpathy",
            "split": "train",
            "max_samples": MAX_VISION_SAMPLES,
        },
        {
            "name": "LAION-400M-Sample",  # Fallback: smaller, actually downloadable
            "repo": "laion/laion400m",
            "split": "train",
            "max_samples": MAX_VISION_SAMPLES,
        }
    ]

    all_vision = []
    image_dir = Path(OUTPUT_DIR) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for cfg in vision_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            # CRITICAL FIX: Use non-streaming with load_dataset for images
            # Or use streaming but decode images properly
            ds = load_dataset(
                cfg["repo"],
                split=cfg["split"],
                streaming=False,  # <-- FIXED: Non-streaming loads actual bytes
                trust_remote_code=True
            )

            # Take subset if too large
            if len(ds) > cfg["max_samples"]:
                ds = ds.shuffle(seed=RANDOM_SEED).select(range(cfg["max_samples"]))

            examples = []
            img_counter = 0

            for item in tqdm(ds, total=min(len(ds), cfg["max_samples"]), desc=f"  {cfg['name']}", ncols=60):
                parsed = parse_vision_item(item, cfg, image_dir, img_counter)
                if parsed:
                    examples.append(parsed)
                    img_counter += 1

            save_path = Path(OUTPUT_DIR) / "vision" / cfg["name"]
            save_both_formats(examples, save_path)
            preview_data(examples, n=2, modality="vision")

            all_vision.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples, {img_counter} images saved")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_vision


def parse_vision_item(item: Dict, cfg: Dict, img_folder: Path, img_idx: int) -> Optional[Dict]:
    """Parse vision item and SAVE the actual image file."""

    img_folder.mkdir(parents=True, exist_ok=True)
    image_path = None

    # METHOD 1: Image is a PIL Image object (non-streaming mode)
    if "image" in item and item["image"] is not None:
        try:
            from PIL import Image
            img_obj = item["image"]

            if isinstance(img_obj, Image.Image):
                img_filename = f"{cfg['name']}_{img_idx:06d}.jpg"
                img_save_path = img_folder / img_filename
                img_obj.save(img_save_path, "JPEG")
                image_path = str(img_save_path.relative_to(OUTPUT_DIR))
                print(f"     🖼️  Saved PIL image: {img_filename}")

            elif isinstance(img_obj, bytes):
                # Raw bytes
                img_filename = f"{cfg['name']}_{img_idx:06d}.jpg"
                img_save_path = img_folder / img_filename
                with open(img_save_path, "wb") as f:
                    f.write(img_obj)
                image_path = str(img_save_path.relative_to(OUTPUT_DIR))
                print(f"     🖼️  Saved bytes image: {img_filename}")

        except Exception as e:
            print(f"     ⚠️  PIL save failed: {e}")

    # METHOD 2: Image is a dict with bytes (datasets Image feature)
    if image_path is None and "image" in item:
        try:
            img_data = item["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
                if img_bytes:
                    img_filename = f"{cfg['name']}_{img_idx:06d}.jpg"
                    img_save_path = img_folder / img_filename
                    with open(img_save_path, "wb") as f:
                        f.write(img_bytes)
                    image_path = str(img_save_path.relative_to(OUTPUT_DIR))
                    print(f"     🖼️  Saved dict-bytes image: {img_filename}")
        except Exception as e:
            print(f"     ⚠️  Dict-bytes save failed: {e}")

    # METHOD 3: Image URL (download it)
    if image_path is None:
        url = item.get("url") or item.get("image_url") or item.get("coco_url") or item.get("flickr_url")
        if url and isinstance(url, str) and url.startswith("http"):
            try:
                response = requests.get(url, timeout=15, stream=True)
                if response.status_code == 200:
                    # Detect extension from content-type or URL
                    ext = ".jpg"
                    content_type = response.headers.get('content-type', '')
                    if 'png' in content_type:
                        ext = ".png"
                    elif 'webp' in content_type:
                        ext = ".webp"

                    img_filename = f"{cfg['name']}_{img_idx:06d}{ext}"
                    img_save_path = img_folder / img_filename

                    with open(img_save_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    image_path = str(img_save_path.relative_to(OUTPUT_DIR))
                    print(f"     🖼️  Downloaded URL image: {img_filename}")
            except Exception as e:
                print(f"     ⚠️  URL download failed: {e}")

    # METHOD 4: filepath reference (copy if exists locally)
    if image_path is None and "filepath" in item:
        fp = item["filepath"]
        if isinstance(fp, str) and os.path.exists(fp):
            img_filename = f"{cfg['name']}_{img_idx:06d}.jpg"
            img_save_path = img_folder / img_filename
            shutil.copy(fp, img_save_path)
            image_path = str(img_save_path.relative_to(OUTPUT_DIR))
            print(f"     🖼️  Copied local image: {img_filename}")

    # Extract text
    text = ""
    response = ""

    if "sentences" in item:
        sents = item["sentences"]
        if isinstance(sents, list) and len(sents) > 0:
            if isinstance(sents[0], dict):
                text = sents[0].get("raw", "") or sents[0].get("text", "")
            else:
                text = str(sents[0])
        response = text
    elif "caption" in item:
        text = item["caption"]
        response = text
    elif "text" in item:
        text = item["text"]
        response = text
    elif "alt" in item:
        text = item["alt"]
        response = text

    if text:
        if image_path:
            return {
                "modality": "vision",
                "source": cfg["name"],
                "image_path": image_path,
                "messages": [
                    {"role": "user", "content": f"<image>\n{text}"},
                    {"role": "assistant", "content": response}
                ]
            }
        else:
            # Text-only fallback
            return {
                "modality": "text",
                "source": cfg["name"],
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response}
                ]
            }

    return None


# ============================================================
# AUDIO DATASETS — FIXED: Actually downloads audio
# ============================================================

def download_audio_datasets():
    """Download audio datasets with ACTUAL audio file downloads."""

    print("\n" + "=" * 60)
    print("🎵 AUDIO DATASETS")
    print("=" * 60)

    audio_configs = [
        {
            "name": "LibriSpeech-clean-100",
            "repo": "openslr/librispeech_asr",
            "config": "clean",
            "split": "train.100",
            "max_samples": MAX_AUDIO_SAMPLES,
            "text_key": "text",
        },
        {
            "name": "CommonVoice-17-en",
            "repo": "mozilla-foundation/common_voice_17_0",
            "config": "en",
            "split": "train",
            "max_samples": MAX_AUDIO_SAMPLES // 2,  # Smaller since it's large
            "text_key": "sentence",
            "requires_auth": True,
        }
    ]

    all_audio = []
    audio_dir = Path(OUTPUT_DIR) / "audio_files"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for cfg in audio_configs:
        print(f"\n  📥 Downloading {cfg['name']}...")

        try:
            load_kwargs = {
                "split": cfg["split"],
                "streaming": False,  # <-- FIXED: Non-streaming for actual audio bytes
                "trust_remote_code": True
            }
            if cfg.get("config"):
                load_kwargs["name"] = cfg["config"]
            if cfg.get("requires_auth"):
                token = os.environ.get("HF_TOKEN")
                if token:
                    load_kwargs["token"] = token
                else:
                    print(f"  ⚠️  Skipping {cfg['name']} — set HF_TOKEN env var")
                    continue

            ds = load_dataset(cfg["repo"], **load_kwargs)

            if len(ds) > cfg["max_samples"]:
                ds = ds.shuffle(seed=RANDOM_SEED).select(range(cfg["max_samples"]))

            examples = []
            audio_counter = 0

            for item in tqdm(ds, total=min(len(ds), cfg["max_samples"]), desc=f"  {cfg['name']}", ncols=60):
                parsed = parse_audio_item(item, cfg, audio_dir, audio_counter)
                if parsed:
                    examples.append(parsed)
                    audio_counter += 1

            save_path = Path(OUTPUT_DIR) / "audio" / cfg["name"]
            save_both_formats(examples, save_path)
            preview_data(examples, n=2, modality="audio")

            all_audio.extend(examples)
            print(f"  ✅ {cfg['name']}: {len(examples):,} examples, {audio_counter} audio files saved")

        except Exception as e:
            print(f"  ❌ Failed {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_audio


def parse_audio_item(item: Dict, cfg: Dict, audio_folder: Path, audio_idx: int) -> Optional[Dict]:
    """Parse audio item and SAVE the actual audio file."""

    audio_folder.mkdir(parents=True, exist_ok=True)

    # Extract text
    text = item.get(cfg["text_key"], "")
    if not text:
        return None

    audio_path = None

    # METHOD 1: Audio is a dict with path + bytes (datasets Audio feature, non-streaming)
    if "audio" in item and item["audio"] is not None:
        try:
            audio_data = item["audio"]

            if isinstance(audio_data, dict):
                # Standard HF datasets Audio format
                orig_path = audio_data.get("path", "")

                # Determine extension
                ext = Path(orig_path).suffix if orig_path else ".wav"
                if not ext or ext == ".":
                    ext = ".wav"

                audio_filename = f"{cfg['name']}_{audio_idx:06d}{ext}"
                audio_save_path = audio_folder / audio_filename

                # Save from array (non-streaming mode gives us the array)
                if "array" in audio_data and "sampling_rate" in audio_data:
                    try:
                        import soundfile as sf
                        import numpy as np

                        array = audio_data["array"]
                        sr = audio_data["sampling_rate"]

                        # Ensure correct shape
                        if len(array.shape) > 1:
                            array = array.squeeze()

                        sf.write(audio_save_path, array, sr)
                        audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                        print(f"     🎵 Saved array audio: {audio_filename}")

                    except ImportError:
                        print("     ⚠️  soundfile not installed. Install: pip install soundfile")
                        # Try scipy as fallback
                        try:
                            from scipy.io import wavfile
                            array = audio_data["array"]
                            if len(array.shape) > 1:
                                array = array.squeeze()
                            # Normalize to int16
                            if array.dtype == np.float32 or array.dtype == np.float64:
                                array = (array * 32767).astype(np.int16)
                            wavfile.write(audio_save_path, audio_data["sampling_rate"], array)
                            audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                            print(f"     🎵 Saved with scipy: {audio_filename}")
                        except ImportError:
                            pass

                # Save from bytes if available
                elif "bytes" in audio_data and audio_data["bytes"]:
                    with open(audio_save_path, "wb") as f:
                        f.write(audio_data["bytes"])
                    audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                    print(f"     🎵 Saved bytes audio: {audio_filename}")

            elif isinstance(audio_data, bytes):
                # Raw bytes
                audio_filename = f"{cfg['name']}_{audio_idx:06d}.wav"
                audio_save_path = audio_folder / audio_filename
                with open(audio_save_path, "wb") as f:
                    f.write(audio_data)
                audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                print(f"     🎵 Saved raw bytes audio: {audio_filename}")

        except Exception as e:
            print(f"     ⚠️  Audio save failed: {e}")

    # METHOD 2: Audio URL
    if audio_path is None:
        url = item.get("audio_url") or item.get("audio_src") or item.get("path", "")
        if isinstance(url, str) and url.startswith("http"):
            try:
                ext = Path(url).suffix or ".wav"
                audio_filename = f"{cfg['name']}_{audio_idx:06d}{ext}"
                audio_save_path = audio_folder / audio_filename

                response = requests.get(url, timeout=15, stream=True)
                if response.status_code == 200:
                    with open(audio_save_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
                    print(f"     🎵 Downloaded URL audio: {audio_filename}")
            except Exception as e:
                print(f"     ⚠️  URL audio download failed: {e}")

    # METHOD 3: Local file path
    if audio_path is None:
        local_path = item.get("path") or item.get("audio_path") or item.get("file")
        if isinstance(local_path, str) and os.path.exists(local_path):
            ext = Path(local_path).suffix or ".wav"
            audio_filename = f"{cfg['name']}_{audio_idx:06d}{ext}"
            audio_save_path = audio_folder / audio_filename
            shutil.copy(local_path, audio_save_path)
            audio_path = str(audio_save_path.relative_to(OUTPUT_DIR))
            print(f"     🎵 Copied local audio: {audio_filename}")

    # If no audio saved, store reference only
    if audio_path is None:
        audio_path = item.get("path", "") or item.get("audio_path", "") or ""

    prompt = "Transcribe this audio"
    if "caption" in cfg["name"].lower():
        prompt = "Describe this audio"

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
# COMBINE & SPLIT
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

    # Save train
    train_jsonl = ready_dir / "train.jsonl"
    train_json = ready_dir / "train.json"
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Save eval
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
    print(f"     📄 {train_jsonl.name}")
    print(f"     📄 {train_json.name}")
    print(f"\n  ✅ Eval:  {len(eval_data):,} examples")
    print(f"     📄 {eval_jsonl.name}")
    print(f"     📄 {eval_json.name}")
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
        print(f"  {key:<45} {count:>8,}")

    print(f"\n  {'TOTAL':<45} {len(all_data):>8,}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Gemma 4-4B Multimodal Dataset Downloader")
    print("   FIXED: Actually downloads images and audio files!")
    print("=" * 60)

    # Check dependencies
    missing = []
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    try:
        import soundfile
    except ImportError:
        try:
            from scipy.io import wavfile
        except ImportError:
            missing.append("soundfile or scipy (pip install soundfile)")

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Images/audio may not save properly.")

    # Check disk space warning
    print(f"\n💾 Output directory: {Path(OUTPUT_DIR).absolute()}")
    print(f"   Estimated size: ~{MAX_VISION_SAMPLES * 0.5:.0f}MB images, ~{MAX_AUDIO_SAMPLES * 2:.0f}MB audio")

    # Clear previous data
    if Path(OUTPUT_DIR).exists():
        response = input(f"\n⚠️  '{OUTPUT_DIR}/' already exists. Delete? [y/N]: ").strip().lower()
        if response == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print("   Cleared existing data.")

    # Download all modalities
    text_data = download_text_datasets()
    vision_data = download_vision_datasets()
    audio_data = download_audio_datasets()

    all_data = text_data + vision_data + audio_data

    if not all_data:
        print("\n❌ No data downloaded. Check errors above.")
        return

    print_summary(all_data)
    combine_and_split(all_data)

    # Verify files exist
    print("\n" + "=" * 60)
    print("🔍 VERIFICATION")
    print("=" * 60)

    img_dir = Path(OUTPUT_DIR) / "images"
    audio_dir = Path(OUTPUT_DIR) / "audio_files"

    if img_dir.exists():
        img_files = list(img_dir.glob("*"))
        print(f"  Images saved: {len(img_files)} files")
        if img_files:
            print(f"  Example: {img_files[0].name}")

    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*"))
        print(f"  Audio saved: {len(audio_files)} files")
        if audio_files:
            print(f"  Example: {audio_files[0].name}")

    print("\n" + "=" * 60)
    print("🎉 DONE! Your datasets are ready:")
    print("   📁 data/ready/train.jsonl  <- Training data")
    print("   📁 data/ready/train.json   <- Human-readable")
    print("   📁 data/images/            <- Downloaded images")
    print("   📁 data/audio_files/       <- Downloaded audio")
    print("=" * 60)


if __name__ == "__main__":
    main()
