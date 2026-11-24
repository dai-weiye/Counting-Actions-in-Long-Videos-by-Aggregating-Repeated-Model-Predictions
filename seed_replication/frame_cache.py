"""Optional temporary pool cache; store the original JPEG data URLs once per frame."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import cv2
import numpy as np


def build(name, folder, videos, design, helpers):
    final=folder/(Path(name).stem+".zip")
    if final.exists():return
    source=next((d/name for d in videos if (d/name).is_file()),None)
    if source is None:raise FileNotFoundError(name)
    with source.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=design["source_sha256"][name]:raise ValueError(name)
    cap=cv2.VideoCapture(str(source))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n!=design["source_frames"][name]:raise ValueError(name)
        temp=final.with_suffix(f".{os.getpid()}.partial")
        indexes=np.linspace(0,n-1,256).astype(int)
        with ZipFile(temp,"w",compression=ZIP_STORED) as z:
            for ix in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix))
                ok,f=cap.read()
                if not ok:raise ValueError(f"{name}: failed {ix}")
                data=helpers.encode_image(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))["image_url"]["url"]
                z.writestr(str(ix),data.encode("ascii"))
            z.writestr("source_sha256",design["source_sha256"][name])
    finally:cap.release()
    os.replace(temp,final)
    print(name,final.stat().st_size,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",required=True,type=Path)
    p.add_argument("--cache-dir",required=True,type=Path)
    p.add_argument("--video-dir",required=True,type=Path,action="append")
    p.add_argument("--inference-protocol",required=True,type=Path)
    p.add_argument("--jobs",type=int,default=1)
    a=p.parse_args();a.cache_dir.mkdir(parents=True,exist_ok=True)
    d=json.loads(a.design.read_text())
    spec=importlib.util.spec_from_file_location("archived_helpers",a.inference_protocol)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(a.jobs) as pool:
        for future in [pool.submit(build,n,a.cache_dir,a.video_dir,d,h) for n in d["filenames"]]:future.result()
