"""Rebuild one archived fourth-call request or parse a saved API response offline."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re


def parse_response(response, options):
    choice = response["choices"][0]
    final = choice["message"].get("content")
    final = final.strip() if isinstance(final, str) else ""
    if re.fullmatch(r"[0-9]+", final) and final in options:
        status, count = "ok", int(final)
    else:
        status, count = ("invalid_final" if final else "blank_final"), None
    return {"status": status, "count": count, "finish_reason": choice.get("finish_reason")}


def request_bytes(protocol, row, video_dir, indices):
    import cv2

    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video_dir / row["video"]))
    content = [{"type": "text", "text": protocol.make_prompt(row["question"], row["candidates"])}]
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {index} of {row['video']}")
            content.append(protocol.encode_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    payload = {"model": "deepseek-flash", "temperature": 0.0, "max_tokens": 8192,
               "thinking": {"type": "enabled"}, "messages": [{"role": "user", "content": content}]}
    return json.dumps(payload, separators=(",", ":")).encode()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="original public 4_count.json with question text")
    parser.add_argument("--item", required=True)
    parser.add_argument("--arm", choices=("uniform", "change"), required=True)
    parser.add_argument("--video-dir", type=Path, help="directory holding the public counting MP4 files")
    parser.add_argument("--frames", type=Path, default=here / "selected_frames.jsonl")
    parser.add_argument("--input-protocol", type=Path, default=here.parent / "inference_protocol.py")
    parser.add_argument("--read-response", type=Path)
    parser.add_argument("--write-request", type=Path)
    args = parser.parse_args()
    if args.write_request is None and args.read_response is None:
        parser.error("Supply --write-request, --read-response, or both")
    rows = {Path(r["video"]).stem: r for r in json.loads(args.annotations.read_text())}
    row = rows[args.item]
    if args.write_request:
        if args.video_dir is None:
            parser.error("--video-dir is required for request reconstruction")
        records = (json.loads(s) for s in args.frames.read_text().splitlines() if s.strip())
        entry = next((r for r in records if r["item_id"] == args.item and r["arm"] == args.arm), None)
        if entry is None:
            parser.error("No terminal acquisition record for this item and arm")
        spec = importlib.util.spec_from_file_location("input_protocol", args.input_protocol)
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        data = request_bytes(protocol, row, args.video_dir, entry["frame_indices"])
        digest = sha256(data).hexdigest()
        if digest != entry["payload_sha256"]:
            raise ValueError(f"Request checksum differs from the acquisition archive: {digest}")
        args.write_request.write_bytes(data)
        print(json.dumps({"item_id": args.item, "arm": args.arm, "payload_sha256": digest}))
    if args.read_response:
        print(json.dumps(parse_response(json.loads(args.read_response.read_text()), row["candidates"])))


if __name__ == "__main__":
    main()
