"""Replay counting input preparation and parsing, or explicitly request one new prediction."""
import argparse
import base64
import io
import json
import os
from urllib import request as http
from urllib.parse import urlsplit
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent


def decode_pool(video, pool_size):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("The supplied video could not be decoded")
        frames, indices = [], []
        for index in np.linspace(0, total - 1, pool_size).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(int(index))
        if not frames:
            raise ValueError("No decodable video frames")
        return frames, indices
    finally:
        capture.release()


def select(frames, budget, policy, seed=None):
    """Return positions in the pool. This preserves the original seed convention."""
    count = len(frames)
    if budget >= count:
        return list(range(count))
    if policy == "change":
        small = [f[::8, ::8].astype(np.float32).mean(axis=-1) for f in frames]
        change = np.zeros(count, dtype=np.float32)
        for i in range(1, count):
            change[i] = np.abs(small[i] - small[i-1]).mean()
        return sorted(map(int, np.argsort(-change)[:budget]))
    if policy == "random":
        base = abs(hash(str(count))) % (2**32)
        s = base if seed is None else (base + 7919*seed) % (2**32)
        return sorted(map(int, np.random.default_rng(s).choice(count, budget, replace=False)))
    if policy == "recency":
        return list(range(count-budget, count))
    if policy == "uniform":
        return np.linspace(0, count-1, budget).round().astype(int).tolist()
    raise ValueError(f"Unknown selection policy: {policy}")


def make_prompt(question, options):
    return (f"Question: {question}\nOptions: {', '.join(options)}\n"
            "Based on these frames, which option is correct? Answer with "
            "the single number from the options and nothing else.")


def parse_response(response, options):
    message = response["choices"][0]["message"]
    text = message.get("content") or ""
    if not text.strip():
        text = message.get("reasoning_content") or ""
    match = re.search(r"\d+", text)
    choices = [int(x) for x in options if x.isdigit()]
    if not match or not choices:
        return None
    return min(choices, key=lambda x: abs(x-int(match.group())))


def encode_image(frame):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="JPEG", quality=90)
    image = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": image}}


class NoRedirect(http.HTTPRedirectHandler):
    """Keep the bearer credential and supplied frames on the configured endpoint."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def submit(payload):
    url = os.environ.get("MAMC_API_URL", "")
    key = os.environ.get("MAMC_API_KEY", "")
    parts = urlsplit(url)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or not key):
        raise ValueError("Set MAMC_API_URL to a full HTTPS endpoint and MAMC_API_KEY to its credential")
    req = http.Request(url, method="POST", data=json.dumps(payload).encode("utf-8"),
                       headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with http.build_opener(NoRedirect()).open(req, timeout=300) as response:
        return json.load(response)


def main():
    config = json.loads((ROOT / "data/runs.json").read_text())
    runs = [k for k, v in config.items() if v["task"] == "4_count"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=runs, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="Public MLVU dev 4_count.json")
    parser.add_argument("--item", required=True, help="Public counting ID, e.g. count_0")
    parser.add_argument("--video", type=Path, help="Matching public MP4 to sample and decode")
    parser.add_argument("--write-request", type=Path, help="Optional local JSON payload including the selected images")
    parser.add_argument("--read-response", type=Path, help="Parse a supplied chat-completions response locally")
    parser.add_argument("--submit", action="store_true",
                        help="Send one new inference request to the explicitly configured HTTPS endpoint")
    args = parser.parse_args()
    if args.submit and args.read_response:
        parser.error("Choose either --submit or --read-response")
    if (args.submit or args.write_request) and not args.video:
        parser.error("--submit and --write-request require --video")
    row = next((r for r in json.loads(args.annotations.read_text())
                if r["video"] == args.item + ".mp4"), None)
    if row is None:
        parser.error("Item not found in the supplied annotations")
    cfg = config[args.run]
    options = row["candidates"]
    prompt = make_prompt(row["question"], options)
    result = {"run": args.run, "item": args.item, "prompt": prompt}
    if args.video:
        frames, indices = decode_pool(args.video, cfg["pool"])
        seed = int(args.run[4:]) if args.run.startswith("seed") else None
        selected = select(frames, cfg["n_frames"], cfg["select"], seed)
        result["source_frame_indices"] = [indices[i] for i in selected]
        if args.write_request or args.submit:
            content = [{"type": "text", "text": prompt}] + [encode_image(frames[i]) for i in selected]
            request = {"model": cfg["model"], "temperature": 0.0,
                       "max_tokens": 8192 if cfg["thinking"] == "enabled" else 64,
                       "thinking": {"type": cfg["thinking"]}, "messages": [{"role": "user", "content": content}]}
            if args.write_request:
                args.write_request.write_text(json.dumps(request))
            if args.submit:
                prediction = parse_response(submit(request), options)
                print(json.dumps({"item_id": args.item, "parsed_count": prediction}))
                return
    if args.read_response:
        result["parsed_count"] = parse_response(json.loads(args.read_response.read_text()), options)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
