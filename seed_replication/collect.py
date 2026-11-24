"""Acquire the frozen seed slots and persist one terminal record per slot."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from threading import Event, local
import time
from zipfile import ZipFile

import cv2
import requests


HTTP = local()


def session(direct):
    if not hasattr(HTTP, "client"):
        HTTP.client = requests.Session()
        HTTP.client.trust_env = not direct
    return HTTP.client


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    tmp = path.with_suffix(".pending")
    with tmp.open("w") as f:
        json.dump(record, f, separators=(",",":"), allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def payload(slot, row, video, helpers, cache=None, source_sha=None):
    content = [{"type": "text", "text": helpers.make_prompt(row["question"], row["candidates"])}]
    if cache is not None and (cache/(slot["item_id"]+".zip")).exists():
        with ZipFile(cache/(slot["item_id"]+".zip")) as z:
            if z.read("source_sha256").decode()!=source_sha: raise ValueError("Mismatched cache source")
            for idx in slot["frame_indices"]:
                content.append({"type":"image_url","image_url":{"url":z.read(str(idx)).decode("ascii")}})
    else:
        cap = cv2.VideoCapture(str(video))
        try:
            for idx in slot["frame_indices"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Frame decode failed at {idx}")
                content.append(helpers.encode_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
    body = {"model":"deepseek-flash", "temperature":0.0, "max_tokens":8192,
            "thinking":{"type":"enabled"}, "messages":[{"role":"user", "content":content}]}
    return json.dumps(body,separators=(",",":")).encode()


def run_slot(slot, annotations, dirs, helpers, run, key, stop, cache, source_sha, direct, expected=None):
    name = f"{slot['item_id']}_g{slot['group']}_c{slot['call']}"
    out = run/"records"/(name+".json")
    mark = run/"attempts"/(name+".json")
    if out.is_file() or stop.is_set():
        return
    row = annotations[slot["item_id"]]
    video = next((d/row["video"] for d in dirs if (d/row["video"]).is_file()),None)
    if video is None:
        raise FileNotFoundError(row["video"])
    start = utc()
    record = {**slot,"started_utc":start, "requested_model":"deepseek-flash",
              "count":None,"finish_reason":None}
    try:
        data = payload(slot,row,video,helpers,cache,source_sha)
    except ValueError:
        write(out,{**record,"status":"decode_error","finished_utc":utc()})
        return
    record["payload_sha256"] = hashlib.sha256(data).hexdigest()
    record["payload_bytes"] = len(data)
    if expected is not None and record["payload_sha256"] != expected[name]:
        raise ValueError(f"Request differs from the independent input reconstruction: {name}")
    hist = json.loads(mark.read_text()) if mark.exists() else {"sha256":record["payload_sha256"],"attempts":[]}
    if hist["sha256"] != record["payload_sha256"]:
        raise ValueError(f"Changed payload: {name}")
    if stop.is_set():
        return
    first_try = len(hist["attempts"])
    for idx in range(first_try,5):
        entry = {"started":utc()}
        hist["attempts"].append(entry)
        write(mark,hist)
        try:
            resp = session(direct).post("https://api.deepseek.com/chat/completions", data=data,
                headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
                allow_redirects=False, timeout=(180,360))
        except requests.RequestException as e:
            entry.update(finished=utc(),error=type(e).__name__)
            write(mark,hist)
            time.sleep(3*(idx+1))
            continue
        entry.update(finished=utc(),http=resp.status_code)
        write(mark,hist)
        if resp.status_code in (401,402,403):
            hist.setdefault("access_pauses",[]).append(hist["attempts"].pop())
            write(mark,hist)
            stop.set()
            raise RuntimeError(f"{name}: HTTP {resp.status_code}, acquisition paused")
        if resp.status_code==429 or 500<=resp.status_code<600:
            time.sleep(3*(idx+1))
            continue
        if resp.status_code!=200:
            record.update(status="http_error",http_status=resp.status_code)
            break
        obj = resp.json()
        choice = obj["choices"][0]
        raw = choice["message"].get("content")
        answer = raw.strip() if isinstance(raw,str) else ""
        ok = answer in row["candidates"] and answer.isascii() and answer.isdecimal()
        record.update(status="ok" if ok else ("invalid_final" if answer else "blank_final"),
                      count=int(answer) if ok else None,final=raw,finish_reason=choice.get("finish_reason"),
                      model=obj.get("model"),fingerprint=obj.get("system_fingerprint"),usage=obj.get("usage"),
                      response_id=obj.get("id"),http_status=resp.status_code)
        break
    else:
        record["status"] = "transport_exhausted"
    record["finished_utc"] = utc()
    write(out,record)
    print(name,record["status"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",required=True,type=Path)
    p.add_argument("--video-dir",required=True,type=Path,action="append")
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--inference-protocol",required=True,type=Path)
    p.add_argument("--expected-hashes", type=Path, help="optional independent SHA-256 map for every scheduled input")
    p.add_argument("--direct",action="store_true",help="connect directly to the official endpoint")
    p.add_argument("--frame-cache-dir",type=Path,help="optional byte-identical pool cache")
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args()
    key=os.environ.get("MAMC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise SystemExit("Set MAMC_API_KEY for the official DeepSeek endpoint")
    for d in ("records","attempts"):
        (a.run/d).mkdir(parents=True,exist_ok=True)
    lock=(a.run/".lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=importlib.util.spec_from_file_location("archived_helpers",a.inference_protocol)
    helpers=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    annotations={Path(r["video"]).stem:r for r in json.loads(a.annotations.read_text())}
    design=json.loads(a.design.read_text())
    opts=[(slot["item_id"],slot["group"],slot["call"]) for slot in design["slots"]]
    if len(opts)!=len(set(opts)): raise ValueError("Duplicate slot in study design")
    expected=json.loads(a.expected_hashes.read_text()) if a.expected_hashes else None
    if expected is not None and any(f"{i}_g{g}_c{c}" not in expected for i,g,c in opts):
        raise ValueError("Missing independently reconstructed input hash")
    stop=Event()
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        fs=[pool.submit(run_slot,s,annotations,a.video_dir,helpers,a.run,key,stop,a.frame_cache_dir,design["source_sha256"][s["item_id"]+".mp4"],a.direct,expected) for s in design["slots"]]
        try:
            for f in concurrent.futures.as_completed(fs):
                f.result()
        except BaseException:
            stop.set()
            for f in fs: f.cancel()
            raise
