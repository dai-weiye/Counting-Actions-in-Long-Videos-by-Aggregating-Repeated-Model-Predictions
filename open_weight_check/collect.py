"""Generate four local open-weight responses per video without using answer labels."""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import struct
import time

import cv2
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import numpy as np
from PIL import Image

from fetch_weights import ROOT, verify

SLOTS = ("U1", "U2", "U3", "C4")


def write_json(path, value):
    temp = path.with_suffix(".pending")
    with temp.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def select_frames(path, change):
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pool = 32 if change else 16
    if count < pool:
        cap.release()
        raise ValueError("Video has fewer source frames than the requested pool")
    indices = np.linspace(0, count - 1, pool).astype(int)
    images = []
    differences = []
    previous = None
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Undecodable source frame {index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if change:
                small = rgb[::8, ::8].astype(np.float32).mean(axis=-1)
                differences.append(0.0 if previous is None else np.abs(small-previous).mean())
                previous = small
            image = Image.fromarray(rgb)
            image.thumbnail((224,224), Image.Resampling.LANCZOS)
            images.append(image)
    finally:
        cap.release()
    keep = sorted(np.argsort(-np.asarray(differences, np.float32))[:16]) if change else range(16)
    input_images = [images[i] for i in keep]
    digest = hashlib.sha256()
    for image in input_images:
        digest.update(struct.pack(">II", *image.size))
        digest.update(image.tobytes())
    return input_images, [int(indices[i]) for i in keep], digest.hexdigest()


def prompt(model, processor, question, options):
    text = (f"Question: {question}\nOptions: {', '.join(options)}\n"
            "Based on these frames, which option is correct? Answer with "
            "the single number from the options and nothing else.")
    return apply_chat_template(processor, model, text, num_images=16)


def execute(model, processor, text, images):
    mx.random.seed(0)
    mx.clear_cache()
    result = generate(model, processor, text, image=images, max_tokens=8, temperature=0.0, verbose=False)
    mx.synchronize()
    return result


def acquire(opts):
    model_files = json.loads((ROOT/"model_files.json").read_text())
    for file, metadata in model_files["files"].items():
        verify(opts.model_dir/file, metadata)
    cohort = json.loads((ROOT/"cohort.json").read_text())
    if len(cohort)!=32 or len({r["video"] for r in cohort})!=32:
        raise ValueError("Expected the locked set of 32 unique videos")
    rows = {q["video"]: (q["question"], q["candidates"]) for q in json.loads(opts.annotations.read_text())}
    opts.run_dir.mkdir(parents=True,exist_ok=True)
    lock=(opts.run_dir/".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record_dir = opts.run_dir/"records"
    record_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    mx.set_memory_limit(4 << 30)
    mx.set_cache_limit(128 << 20)
    model, processor = load(str(opts.model_dir))
    cfg = load_config(str(opts.model_dir))
    if opts.smoke:
        images = [Image.new("RGB",(224,168),(16*i, 16*i, 16*i)) for i in range(16)]
        text = prompt(cfg, processor, "How many people appear?", ["0","1","2","3"])
        result = execute(model, processor, text, images)
        print("Initialization smoke test succeeded, peak working GPU bytes",mx.get_peak_memory(),
              "prompt tokens",result.prompt_tokens,"finish",result.finish_reason,flush=True)
        return
    settings = {"model_repo":model_files["repo_id"],"revision":model_files["revision"],
                "device":mx.device_info()["device_name"],"machine":platform.machine(),
                "os_version":platform.mac_ver()[0],"gpu_architecture":mx.device_info()["architecture"],
                "python":platform.python_version(),"packages":{dist:version(dist) for dist in (
                    "numpy","Pillow","mlx","mlx-vlm","transformers","opencv-python","requests")},
                "temperature":0.0,"mlx_seed_per_call":0,"max_new_tokens":8,"visual_budget":16,
                "max_image_side":224,"model_dir_manifest_sha256":hashlib.sha256((ROOT/"model_files.json").read_bytes()).hexdigest()}
    settings_path=opts.run_dir/"settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError("Runtime settings changed across collection")
    write_json(settings_path, settings)
    if opts.item is not None and opts.item not in {Path(r["video"]).stem for r in cohort}:
        raise ValueError("Item is outside the frozen panel")
    for n, cohort_row in enumerate(cohort, 1):
        name=cohort_row["video"]
        item=Path(name).stem
        if opts.item is not None and item!=opts.item:
            continue
        if all((record_dir/f"{item}_{s}.json").exists() for s in SLOTS):
            continue
        video=opts.video_dir/name
        with video.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=cohort_row["source_sha256"]:
                raise ValueError(f"Unrecognized video content for {name}")
        question,options=rows[name]
        text=prompt(cfg,processor,question,options)
        visual={}
        for slot in SLOTS:
            output=record_dir/f"{item}_{slot}.json"
            if output.exists():
                saved=json.loads(output.read_text())
                if (saved["item_id"],saved["slot"])!=(item,slot):raise ValueError("Record key mismatch")
                continue
            change = slot=="C4"
            if change not in visual:
                visual[change]=select_frames(video,change)
            images,frames,img_sha=visual[change]
            start=datetime.now(timezone.utc).isoformat()
            begin=time.monotonic()
            result=execute(model,processor,text,images)
            raw=result.text or ""
            value=raw.strip()
            ok=value in options and value.isascii() and value.isdecimal()
            record={"item_id":item,"slot":slot,"source_indices":frames,"rgb_sha256":img_sha,
                    "formatted_prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),
                    "image_dimensions":[list(image.size) for image in images],"final":raw,
                    "count":int(value) if ok else None,
                    "status":"ok" if ok else ("invalid_final" if value else "blank_final"),
                    "finish_reason":result.finish_reason,"generated_tokens":int(result.generation_tokens),
                    "prompt_tokens":int(result.prompt_tokens),"started_utc":start,
                    "elapsed_s":round(time.monotonic()-begin,4)}
            write_json(output,record)
            print(f"{n}/32 {item} {slot} {record['status']}, peak {mx.get_peak_memory()/2**30:.2f} GiB",flush=True)
            del result
            mx.clear_cache()
        del visual
        gc.collect()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--video-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--item",help="Replay the four slots of one named cohort item")
    parser.add_argument("--smoke",action="store_true",help="Test only on synthetic neutral frames")
    acquire(parser.parse_args())
