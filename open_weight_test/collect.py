"""Run a pinned 16-frame Qwen model on the 60 official test videos."""

import argparse
import fcntl
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import load_config

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "open_weight_check"
sys.path.insert(0, str(BASE))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "open_weight_dev_collection", BASE / "collect.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from fetch_weights import verify


def acquire(args):
    roster = json.loads((HERE / "cohort.json").read_text())
    if len(roster) != 60 or len({r["id"] for r in roster}) != 60:
        raise ValueError("Expected 60 official questions")
    manifest_path = BASE / "model_files.json"
    manifest = json.loads(manifest_path.read_text())
    for name, meta in manifest["files"].items():
        verify(args.model_dir / name, meta)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record_dir = args.run_dir / "records"
    record_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    mx.set_memory_limit(4 << 30)
    mx.set_cache_limit(128 << 20)
    model, processor = load(str(args.model_dir))
    cfg = load_config(str(args.model_dir))
    settings = {
        "revision": manifest["revision"],
        "model_repo": manifest["repo_id"],
        "device": mx.device_info()["device_name"],
        "architecture": mx.device_info()["architecture"],
        "os_version": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": {
            d: version(d)
            for d in (
                "numpy",
                "Pillow",
                "mlx",
                "mlx-vlm",
                "transformers",
                "opencv-python",
            )
        },
        "temperature": 0.0,
        "seed_reset_per_generation": 0,
        "max_new_tokens": 8,
        "model_file_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "cohort_sha256": hashlib.sha256(
            (HERE / "cohort.json").read_bytes()
        ).hexdigest(),
    }
    settings_path = args.run_dir / "runtime.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Runtime changed across collection")
    base.write_json(settings_path, settings)
    if args.item and args.item not in {r["id"] for r in roster}:
        raise ValueError("Requested item is outside the complete official split")
    for n, row in enumerate(roster, 1):
        item = row["id"]
        if args.item and args.item != item:
            continue
        if all((record_dir / f"{item}_{s}.json").is_file() for s in base.SLOTS):
            continue
        video = args.video_dir / row["video"]
        with video.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != row["source_sha256"]:
                raise ValueError(f"Video checksum mismatch for {item}")
        choices = row["options"]
        if len(set(choices)) != 6 or not all(
            s.isascii() and s.isdecimal() for s in choices
        ):
            raise ValueError(f"Invalid offered options for {item}")
        text = base.prompt(cfg, processor, row["question"], choices)
        views = {}
        for slot in base.SLOTS:
            output = record_dir / f"{item}_{slot}.json"
            if output.exists():
                saved = json.loads(output.read_text())
                if (saved["item_id"], saved["slot"]) != (item, slot):
                    raise ValueError("Existing record has the wrong key")
                continue
            change = slot == "C4"
            if change not in views:
                views[change] = base.select_frames(video, change)
            images, indices, digest = views[change]
            start = datetime.now(timezone.utc).isoformat()
            result = base.execute(model, processor, text, images)
            raw = result.text or ""
            value = raw.strip()
            ok = value in choices
            record = {
                "item_id": item,
                "slot": slot,
                "source_indices": indices,
                "rgb_sha256": digest,
                "formatted_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "image_dimensions": [list(im.size) for im in images],
                "final": raw,
                "count": int(value) if ok else None,
                "status": "ok" if ok else ("invalid_final" if value else "blank_final"),
                "finish_reason": result.finish_reason,
                "generated_tokens": int(result.generation_tokens),
                "prompt_tokens": int(result.prompt_tokens),
                "started_utc": start,
            }
            base.write_json(output, record)
            print(n, item, slot, record["status"], flush=True)
            del result
            mx.clear_cache()
        del views
        gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--item")
    acquire(p.parse_args())
