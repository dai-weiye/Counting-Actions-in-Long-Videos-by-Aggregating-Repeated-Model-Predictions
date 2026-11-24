"""Resume the frozen test protocol and reject all outgoing payloads absent from its input lock."""

import argparse
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--input-lock-dir", type=Path, required=True)
    ap.add_argument("--score-lock", type=Path, default=ROOT / "analysis_lock.json")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    input_lock = read_json(args.input_lock_dir / "input_lock.json")
    scoring_lock = read_json(args.score_lock)["sha256"]
    checked = [
        (args.questions, input_lock["questions"]),
        (
            args.input_lock_dir / "planned_slots.jsonl",
            input_lock["planned_slots.jsonl"],
        ),
        (
            args.input_lock_dir / "planned_frame_inputs.jsonl",
            input_lock["planned_frame_inputs.jsonl"],
        ),
        (args.protocol_root / "fourth_call/collect.py", input_lock["sampler"]),
        (args.protocol_root / "inference_protocol.py", input_lock["original_protocol"]),
    ]
    for name, digest in scoring_lock.items():
        source = (
            args.protocol_root if name.startswith("fourth_call/") else ROOT
        ) / name
        checked.append((source, digest))
    for path, expected in checked:
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source mismatch: {path.name}")
    planned = {
        (r["id"], r["slot"]): r["payload_sha256"]
        for r in read_lines(args.input_lock_dir / "planned_slots.jsonl")
    }
    id_by_video = {Path(q["video"]).stem: q["id"] for q in read_json(args.questions)}
    completed = set()
    for path in args.run_dir.glob("*/fourth_calls.jsonl"):
        for row in read_lines(path):
            key = (id_by_video[row["item_id"]], path.parent.name)
            if key in completed or row["payload_sha256"] != planned[key]:
                raise ValueError("Completed slot mismatch")
            completed.add(key)
    if len(planned) != 240:
        raise ValueError("Incomplete fixed cohort")
    allowed = {v for k, v in planned.items() if k not in completed}
    print(
        f"Verified {len(planned)} frozen slots and {len(completed)} actual terminal receipts.",
        flush=True,
    )
    if args.check_only:
        return
    original_post = requests.post

    def checked_post(url, **kwargs):
        if (
            url != "https://api.deepseek.com/chat/completions"
            or sha256(kwargs["data"]).hexdigest() not in allowed
        ):
            raise ValueError("Outgoing request differs from the frozen test inputs")
        return original_post(url, **kwargs)

    requests.post = checked_post
    spec = importlib.util.spec_from_file_location(
        "blind_test_collector", ROOT / "collect_test.py"
    )
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    try:
        collector.run(args)
    finally:
        requests.post = original_post


if __name__ == "__main__":
    main()
