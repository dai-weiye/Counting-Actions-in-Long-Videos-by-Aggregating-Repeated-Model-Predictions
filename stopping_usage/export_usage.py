"""Export observed token counters from the 240 archived official-test API responses."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path


SLOTS = ("U1", "U2", "U3", "C4")
COUNTERS = ("prompt_tokens", "completion_tokens", "total_tokens")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def export(raw, release, collector, target):
    roster = json.loads((release / "items.json").read_text(encoding="utf-8"))
    video_ids = {Path(row["video"]).stem: row["id"] for row in roster}
    calls = {(r["id"], r["slot"]): r for r in read_lines(release / "calls.jsonl")}
    inputs = {(r["id"], r["slot"]): r["payload_sha256"] for r in read_lines(release / "planned_slots.jsonl")}
    expected = set(product(video_ids.values(), SLOTS))
    if len(expected) != 240 or set(calls) != expected or set(inputs) != expected:
        raise ValueError("The released cohort, final responses and planned requests must agree on 240 slots")

    outputs = {}
    source_hashes = {}
    for slot in SLOTS:
        path = raw / slot / "fourth_calls.jsonl"
        source_hashes[slot] = digest(path)
        for row in read_lines(path):
            video = video_ids[row["item_id"]]
            key = (video, slot)
            if key in outputs or row["http_status"] != 200 or row["payload_sha256"] != inputs[key]:
                raise ValueError(f"Duplicate or mismatched receipt for {video}/{slot}")
            if any(row[k] != calls[key][k] for k in ("status", "count", "finish_reason")):
                raise ValueError(f"Response does not match the published evaluation for {video}/{slot}")
            usage = {k: row["usage"][k] for k in COUNTERS}
            if any(type(n) is not int or n < 0 for n in usage.values()):
                raise ValueError(f"Expected nonnegative integer usage counters for {video}/{slot}")
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                raise ValueError(f"Inconsistent total for {video}/{slot}")
            outputs[key] = dict(id=video, slot=slot, payload_sha256=inputs[key], **usage)
    if set(outputs) != expected:
        raise ValueError("Incomplete receipt ledger")

    target.mkdir(parents=True, exist_ok=True)
    file = target / "official_api_usage.jsonl"
    with file.open("w", encoding="utf-8", newline="\n") as stream:
        for item in roster:
            for slot in SLOTS:
                stream.write(json.dumps(outputs[item["id"], slot], separators=(",", ":")) + "\n")
    provenance = {
        "description": "Observed usage replay under prefixes of four archived responses per video. Each replay sums the counters for the retained slots.",
        "questions": len(roster),
        "slots": len(outputs),
        "totals": {name: sum(row[name] for row in outputs.values()) for name in COUNTERS},
        "usage_jsonl_sha256": digest(file),
        "source_receipt_files_sha256_by_slot": source_hashes,
        "release_files_sha256": {
            name: digest(release / name) for name in (
                "items.json", "calls.jsonl", "planned_slots.jsonl", "collect_test.py", "resume_locked_test.py"
            )
        },
        "receipt_collector_sha256": digest(collector),
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Directory with a fourth_calls.jsonl receipt file for each slot")
    parser.add_argument("--release", required=True, type=Path, help="Completed official-test release directory")
    parser.add_argument("--collector", required=True, type=Path, help="Shared collector that saved the response usage counters")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the anonymous ledger and its provenance")
    args = parser.parse_args()
    export(args.raw, args.release, args.collector, args.out)
