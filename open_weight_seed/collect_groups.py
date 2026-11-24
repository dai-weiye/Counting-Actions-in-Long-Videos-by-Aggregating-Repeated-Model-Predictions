"""Run the locked 16-frame seed panel with the shared, verified local model."""
import argparse
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import struct
import sys
import time

import cv2
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import load_config
from PIL import Image


def read_pixels(path,indices):
    cap=cv2.VideoCapture(str(path))
    result={}
    try:
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<=max(indices):
            raise ValueError(f"Unexpected source length for {path.name}")
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,frame=cap.read()
            if not ok:raise ValueError(f"Frame {idx} cannot be decoded in {path.name}")
            rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            rgb.thumbnail((224,224),Image.Resampling.LANCZOS)
            result[idx]=rgb
    finally:cap.release()
    return result


def rgb_digest(images):
    h=hashlib.sha256()
    for im in images:
        h.update(struct.pack(">II",*im.size))
        h.update(im.tobytes())
    return h.hexdigest()


def run(a):
    base=a.reference_dir.resolve()
    sys.path.insert(0,str(base))
    spec=importlib.util.spec_from_file_location("validated_open_model",base/"collect.py")
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    manifest=json.loads((base/"model_files.json").read_text())
    if manifest["revision"]!="01af461cdb9574acc09084a0ef94e216e142b085": raise ValueError("Unexpected model revision")
    for name,meta in manifest["files"].items():ref.verify(a.model_dir/name,meta)
    d=json.loads(a.design.read_text()); design_hash=hashlib.sha256(a.design.read_bytes()).hexdigest()
    slots={(s["item_id"],s["group"],s["call"]):s for s in d["slots"]}
    if len(slots)!=len(d["slots"]) or len(slots)!=768:raise ValueError("The six-group design requires 768 slots")
    rows={r["video"]:(r["question"],r["candidates"]) for r in json.loads(a.annotations.read_text())}
    a.run_dir.mkdir(exist_ok=True,parents=True)
    record_dir=a.run_dir/"records";record_dir.mkdir(exist_ok=True)
    lock=(a.run_dir/".gpu_job_lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cv2.setNumThreads(1);mx.set_memory_limit(4<<30);mx.set_cache_limit(128<<20)
    model,processor=load(str(a.model_dir)); cfg=load_config(str(a.model_dir))
    settings={"model_repo":manifest["repo_id"],"revision":manifest["revision"],"device":mx.device_info()["device_name"],
              "machine":platform.machine(),"os_version":platform.mac_ver()[0],"gpu_architecture":mx.device_info()["architecture"],
              "python":platform.python_version(),"packages":{k:version(k) for k in ("numpy","Pillow","mlx","mlx-vlm","transformers","opencv-python","requests")},
              "temperature":0.0,"mlx_seed_per_call":0,"max_new_tokens":8,"visual_budget":16,"max_image_side":224,
              "model_dir_manifest_sha256":hashlib.sha256((base/"model_files.json").read_bytes()).hexdigest(),
              "seed_design_sha256":design_hash}
    target=a.run_dir/"settings.json"
    if target.exists() and json.loads(target.read_text())!=settings:raise ValueError("Local runtime changed across collection")
    ref.write_json(target,settings)
    for name in d["filenames"]:
        item=Path(name).stem
        todo=[(g,c,record_dir/f"{item}_g{g}_c{c}.json") for g in range(1,7) for c in range(1,5)]
        todo=[v for v in todo if not v[2].is_file()]
        if not todo:continue
        f=a.video_dir/name
        with f.open("rb") as source:
            if hashlib.file_digest(source,"sha256").hexdigest()!=d["source_sha256"][name]:raise ValueError(name+" SHA-256")
        required={i for g,c,_ in todo for i in slots[item,g,c]["frame_indices"]}
        cache=read_pixels(f,required)
        question,options=rows[name];text=ref.prompt(cfg,processor,question,options)
        for g,c,out in todo:
            indices=slots[item,g,c]["frame_indices"];images=[cache[i] for i in indices]
            start=datetime.now(timezone.utc).isoformat(); tick=time.monotonic()
            value=ref.execute(model,processor,text,images)
            final=value.text or "";t=final.strip()
            ok=t in options and t.isascii() and t.isdecimal()
            ref.write_json(out,{"item_id":item,"group":g,"call":c,"frame_indices":indices,
                         "rgb_sha256":rgb_digest(images),"formatted_prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),
                         "image_dimensions":[list(im.size) for im in images],"final":final,"count":int(t) if ok else None,
                         "status":"ok" if ok else ("invalid_final" if t else "blank_final"),
                         "finish_reason":value.finish_reason,"generated_tokens":int(value.generation_tokens),
                         "prompt_tokens":int(value.prompt_tokens),"started_utc":start,"elapsed_s":round(time.monotonic()-tick,4)})
            print(item,g,c,"completed",flush=True);del value;mx.clear_cache()
        del cache;gc.collect()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("reference-dir","model-dir","annotations","video-dir","design","run-dir"):
        p.add_argument("--"+n,required=True,type=Path)
    run(p.parse_args())
