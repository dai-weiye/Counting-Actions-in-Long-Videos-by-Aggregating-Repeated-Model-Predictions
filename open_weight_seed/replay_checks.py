"""Cold-load the pinned checkpoint and replay two predetermined boundary slots."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import load_config
import numpy as np
from PIL import Image
from collect_groups import rgb_digest


def replay(a):
    base=a.reference_dir.resolve();sys.path.insert(0,str(base))
    sp=importlib.util.spec_from_file_location("validated_local_runtime",base/"collect.py")
    ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
    m=json.loads((base/"model_files.json").read_text())
    for f,meta in m["files"].items():ref.verify(a.model_dir/f,meta)
    cv2.setNumThreads(1);mx.set_memory_limit(4<<30);mx.set_cache_limit(128<<20)
    model,processor=load(str(a.model_dir));cfg=load_config(str(a.model_dir))
    d=json.loads(a.design.read_text())
    ann={r["video"]:r for r in json.loads(a.annotations.read_text())}
    preds={(r["item_id"],r["group"],r["call"]):r for r in map(json.loads,a.predictions.read_text().splitlines())}
    if len(preds)!=768:raise ValueError("The full local ledger must close before cold replay")
    results=[]
    for name,g,c in ((d["filenames"][0],1,1),(d["filenames"][-1],6,4)):
        q=ann[name];item=Path(name).stem;r=preds[item,g,c]
        pool=np.linspace(0,d["source_frames"][name]-1,256).astype(int)
        target=set(r["frame_indices"]);images={}
        with (a.video_dir/name).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==d["source_sha256"][name]
        cap=cv2.VideoCapture(str(a.video_dir/name))
        try:
            for ix in pool:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix));ok,pix=cap.read()
                if not ok:raise ValueError(f"Frame decode failed: {name} {ix}")
                if int(ix) in target:
                    im=Image.fromarray(cv2.cvtColor(pix,cv2.COLOR_BGR2RGB));im.thumbnail((224,224),Image.Resampling.LANCZOS)
                    images[int(ix)]=im
        finally:cap.release()
        imgs=[images[k] for k in r["frame_indices"]]
        assert rgb_digest(imgs)==r["rgb_sha256"]
        prompt=ref.prompt(cfg,processor,q["question"],q["candidates"])
        assert hashlib.sha256(prompt.encode()).hexdigest()==r["formatted_prompt_sha256"]
        response=ref.execute(model,processor,prompt,imgs)
        value=response.text or ""
        results.append({"item_id":item,"group":g,"call":c,"rgb_sha256_matched":True,"prompt_sha256_matched":True,
                        "same_final_text":value==r["final"],"same_finish_reason":response.finish_reason==r["finish_reason"],
                        "same_generated_tokens":int(response.generation_tokens)==r["generated_tokens"]})
    a.out.write_text(json.dumps(results,indent=2)+"\n")
    for r in results:print(r["item_id"],r["group"],r["call"],r["same_final_text"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("reference-dir","model-dir","video-dir","annotations","design","predictions","out"):p.add_argument("--"+k,type=Path,required=True)
    replay(p.parse_args())
