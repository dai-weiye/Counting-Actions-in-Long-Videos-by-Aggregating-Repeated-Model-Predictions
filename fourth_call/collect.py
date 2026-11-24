"""Collect one new paired comparison, using identical prompts and JPEG encoding in both arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://api.deepseek.com/chat/completions"
SPEC = importlib.util.spec_from_file_location(
    "input_protocol", HERE.parent / "inference_protocol.py")
original = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(original)
ARMS = ("uniform", "change")
IO_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(".pending")
    with tmp.open("w") as out:
        json.dump(value, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def append_json(path, value):
    with IO_LOCK, path.open("a") as out:
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), file=out)
        out.flush()
        os.fsync(out.fileno())


def sample(path, arm):
    pool = 128 if arm == "uniform" else 256
    cap = cv2.VideoCapture(str(path))
    images, indices, differences = [], [], []
    previous = None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < pool:
            raise ValueError("short_or_invalid_video")
        for idx in np.linspace(0, total - 1, pool).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise ValueError("frame_decode_failed")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(original.encode_image(rgb))
            indices.append(int(idx))
            if arm == "change":
                small = rgb[::8, ::8].astype(np.float32).mean(axis=-1)
                differences.append(0.0 if previous is None else np.abs(small-previous).mean())
                previous = small
    finally:
        cap.release()
    selected = range(128) if arm == "uniform" else sorted(np.argsort(-np.asarray(differences, np.float32))[:128])
    return [images[i] for i in selected], [indices[i] for i in selected]


def collect_arm(row, video, arm, key, run, done, stop, expected_payload_sha256=None, timeout=(30, 360)):
    item = Path(row["video"]).stem
    if (item, arm) in done or stop.is_set():
        return
    mark = run / "inflight" / f"{item}-{arm}.json"
    previous = json.loads(mark.read_text()) if mark.exists() else None
    prompt = original.make_prompt(row["question"], row["candidates"])
    images, indices = sample(video, arm)
    payload = {"model": "deepseek-flash", "temperature": 0.0, "max_tokens": 8192,
               "thinking": {"type": "enabled"}, "messages": [{"role": "user", "content":
               [{"type": "text", "text": prompt}, *images]}]}
    data = json.dumps(payload, separators=(",", ":")).encode()
    del images, payload
    record = {"item_id": item, "arm": arm, "requested_model": "deepseek-flash", "frame_indices": indices,
              "payload_sha256": hashlib.sha256(data).hexdigest(), "payload_bytes": len(data)}
    if expected_payload_sha256 is not None and record["payload_sha256"] != expected_payload_sha256:
        raise ValueError(f"Reconstructed request differs for {item} {arm}")
    if stop.is_set():
        return
    first_attempt = 1
    if previous:
        if previous["payload_sha256"] != record["payload_sha256"]:
            raise RuntimeError(f"Recovery payload changed for {item} {arm}")
        first_attempt = previous["attempt"] + 1
    for attempt in range(first_attempt, 6):
        started = now()
        write_json(mark, {**record, "attempt": attempt, "started_utc": started})
        try:
            response = requests.post(ENDPOINT, data=data, headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + key}, timeout=timeout, allow_redirects=False)
        except requests.RequestException as exc:
            append_json(run / "attempts.jsonl", {"item_id": item, "arm": arm, "attempt": attempt,
                        "started_utc": started, "finished_utc": now(), "transport_error": type(exc).__name__})
            print(f"{item} {arm}: {type(exc).__name__}, attempt {attempt} returned no response", flush=True)
            time.sleep(min(30, 4 * attempt))
            continue
        audit = {"item_id": item, "arm": arm, "attempt": attempt, "started_utc": started,
                 "finished_utc": now(), "http_status": response.status_code,
                 "request_id": response.headers.get("x-request-id")}
        append_json(run / "attempts.jsonl", audit)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(4 * attempt)
            continue
        if not response.ok and response.status_code in (401, 402, 403):
            mark.unlink(missing_ok=True)
            stop.set()
            raise RuntimeError(f"{item} {arm}: HTTP {response.status_code}; acquisition paused")
        if not response.ok:
            record.update(audit, status="http_error", count=None, final=None, finish_reason=None)
            append_json(run / "fourth_calls.jsonl", record)
            mark.unlink(missing_ok=True)
            print(f"{item} {arm}: HTTP {response.status_code}", flush=True)
            return
        obj = response.json()
        choice = obj["choices"][0]
        final = choice["message"].get("content")
        final = final if isinstance(final, str) else ""
        reported = final.strip()
        accepted = re.fullmatch(r"[0-9]+", reported) is not None and reported in row["candidates"]
        record.update(audit, status="ok" if accepted else ("blank_final" if not reported else "invalid_final"),
                      count=int(reported) if accepted else None, final=final,
                      finish_reason=choice.get("finish_reason"), response_id=obj.get("id"),
                      model_served=obj.get("model"), fingerprint=obj.get("system_fingerprint"),
                      usage=obj.get("usage"), provider_created=obj.get("created"))
        append_json(run / "fourth_calls.jsonl", record)
        mark.unlink(missing_ok=True)
        print(f"{item} {arm}: {record['status']}", flush=True)
        return
    record.update(status="transport_exhausted", count=None, final=None, finish_reason=None,
                  finished_utc=now())
    append_json(run / "fourth_calls.jsonl", record)
    mark.unlink(missing_ok=True)
    print(f"{item} {arm}: no response after five attempts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="public counting annotation JSON")
    parser.add_argument("--video-dir", type=Path, action="append", required=True, help="MP4 directory, repeatable")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("holdout", "all"), default="holdout")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Number of planned pairs to visit, in fixed order; resume with no limit")
    args = parser.parse_args()
    key = os.getenv("MAMC_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set MAMC_API_KEY or DEEPSEEK_API_KEY for the project's official DeepSeek endpoint")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "inflight").mkdir(exist_ok=True)
    lock = (run / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run / "fourth_calls.jsonl"
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            keypair = (r["item_id"], r["arm"])
            if keypair in done:
                raise RuntimeError(f"Duplicate terminal record {keypair}")
            done.add(keypair)
        for item, arm in done:
            (run / "inflight" / f"{item}-{arm}.json").unlink(missing_ok=True)
    rows = sorted(json.loads(args.annotations.read_text()), key=lambda r: r["video"])
    order = np.random.default_rng(0).permutation(len(rows))
    planned = list(order[len(rows)//2:])
    if args.scope == "all":
        planned.extend(order[:len(rows)//2])
    cv2.setNumThreads(1)
    roots = args.video_dir
    stop = threading.Event()

    def process(position, index):
        row = rows[index]
        video = next((root / row["video"] for root in roots if (root / row["video"]).is_file()), None)
        if video is None:
            print(f"{row['video']}: input pending", flush=True)
            return
        for arm in ARMS if position % 2 == 0 else ARMS[::-1]:
            collect_arm(row, video, arm, key, run, done, stop)

    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(process, pos, i) for pos, i in enumerate(planned[:args.limit])]
        try:
            for f in as_completed(futures):
                f.result()
        except BaseException:
            stop.set()
            for f in futures:
                f.cancel()
            raise


if __name__ == "__main__":
    main()
