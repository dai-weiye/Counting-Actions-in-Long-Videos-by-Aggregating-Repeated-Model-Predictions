"""Derive the frozen local 16-frame inputs from the original 128-frame seed design."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def derive(parent_bytes):
    sha=hashlib.sha256(parent_bytes).hexdigest()
    if sha!="d7f81f6a26ecb6fe567312564ef9f176f4fc351198943e837c760273fb437f8c":
        raise ValueError("Unexpected parent input design")
    d=json.loads(parent_bytes)
    sub=np.linspace(0,127,16).astype(int).tolist()
    return {**d,"parent_design_sha256":sha,"subsample_positions_in_128":sub,
            "stream_positions":[[m[i] for i in sub] for m in d["stream_positions"]],
            "slots":[{**s,"frame_indices":[s["frame_indices"][i] for i in sub]} for s in d["slots"]]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent-design",type=Path)
    p.add_argument("--annotations",type=Path);p.add_argument("--video-dir",type=Path);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-design-script",type=Path,default=Path(__file__).with_name("source_seed_design.py"))
    a=p.parse_args()
    if a.parent_design:raw=a.parent_design.read_bytes()
    else:
        if not a.annotations or not a.video_dir:p.error("Set --parent-design or --annotations and --video-dir")
        sp=importlib.util.spec_from_file_location("seed_source",a.source_design_script)
        mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
        parent=mod.make_design(json.loads(a.annotations.read_text()),[a.video_dir])
        raw=(json.dumps(parent,separators=(",",":"))+"\n").encode()
    text=json.dumps(derive(raw),separators=(",",":"))+"\n"
    if a.out.is_file() and a.out.read_text()!=text:
        raise ValueError("Recomputed design differs from the existing file")
    a.out.write_text(text)
