"""Recreate and verify one recorded seed request without sending it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from collect import payload


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,required=True)
    p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--item",required=True)
    p.add_argument("--group",type=int,required=True)
    p.add_argument("--call",type=int,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--video-dir",type=Path,required=True)
    p.add_argument("--inference-protocol",type=Path,required=True)
    a=p.parse_args()
    def same(r): return (r["item_id"],r["group"],r["call"])==(a.item,a.group,a.call)
    slot=next(s for s in json.loads(a.design.read_text())["slots"] if same(s))
    record=next(r for r in map(json.loads,a.counts.read_text().splitlines()) if same(r))
    if not record["payload_sha256"]:raise ValueError("No request was encoded for this slot")
    row=next(r for r in json.loads(a.annotations.read_text()) if r["video"]==a.item+".mp4")
    spec=importlib.util.spec_from_file_location("archived_helpers",a.inference_protocol)
    helpers=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    b=payload(slot,row,a.video_dir/row["video"],helpers)
    if hashlib.sha256(b).hexdigest()!=record["payload_sha256"]:raise ValueError("Payload checksum mismatch")
    print("Verified",a.item,a.group,a.call,len(b),"bytes")
