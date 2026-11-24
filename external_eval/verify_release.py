"""Verify the archived test responses, their frozen inputs and the 160-response checkpoint."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return list(map(json.loads, path.read_text().splitlines()))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_key(rows, field):
    index = {(r["id"], r[field]): r for r in rows}
    assert len(rows) == len(index), "Duplicate observation key"
    return index


def verify(root):
    lock = load(root / "input_lock.json")
    for name in ("planned_slots.jsonl", "planned_frame_inputs.jsonl"):
        assert sha256(root / name) == lock[name]
    items = load(root / "items.json")
    cohort = {i["id"] for i in items}
    expected = {(i, slot) for i in cohort for slot in ("U1", "U2", "U3", "C4")}
    slots = by_key(lines(root / "planned_slots.jsonl"), "slot")
    calls = by_key(lines(root / "calls.jsonl"), "slot")
    assert set(calls) == set(slots) == expected and len(slots) == 240
    before = by_key(lines(root / "checkpoint_at_pause" / "calls_received.jsonl"), "slot")
    pending = by_key(lines(root / "checkpoint_at_pause" / "unreceived_slots.jsonl"), "slot")
    assert (len(before), len(pending)) == (160, 80)
    assert not before.keys() & pending.keys() and before.keys() | pending.keys() == expected
    for key, old in before.items():
        assert old == dict(calls[key], payload_sha256=slots[key]["payload_sha256"])
    for key, old in pending.items():
        assert old["payload_sha256"] == slots[key]["payload_sha256"]
    inputs = by_key(lines(root / "planned_frame_inputs.jsonl"), "arm")
    frames = by_key(lines(root / "selected_frames.jsonl"), "arm")
    assert inputs.keys() == frames.keys() and len(frames) == 120
    video = {r["id"]: r for r in load(root / "test_video_digests.json")}
    sources = load(root / "video_sources.json")["videos"]
    assert set(video) == cohort and len(cohort) == 60
    for key, row in frames.items():
        f = inputs[key]
        assert row["frame_indices"] == f["frame_indices"]
        assert row["payload_sha256"] == f["payload_sha256"]
        v = video[key[0]]
        assert (v["video"], v["sha256"]) == (f["video"], f["video_sha256"])
        assert v["bytes"] == sources[v["video"]]["bytes"]
    for (i, s), row in slots.items():
        assert row["payload_sha256"] == frames[i, "change" if s == "C4" else "uniform"]["payload_sha256"]
    dev = load(root / "dev_video_digests.json")
    assert len(dev) == 206
    for field in ("id", "sha256", "video"):
        assert {r[field] for r in video.values()}.isdisjoint(r[field] for r in dev)
    return Counter(r["status"] for r in calls.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    result = verify(parser.parse_args().root)
    print("Verified 240 slots, all frozen inputs, and 160 unchanged checkpoint responses.")
    print(json.dumps(dict(sorted(result.items()))))
