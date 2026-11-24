"""Freeze a complete test prediction ledger before scoring the labels."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")
FIELDS = (
    "item_id",
    "slot",
    "source_indices",
    "rgb_sha256",
    "formatted_prompt_sha256",
    "image_dimensions",
    "status",
    "count",
    "final",
    "finish_reason",
    "generated_tokens",
    "prompt_tokens",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(run_dir, out):
    roster = {r["id"]: r for r in json.loads((HERE / "cohort.json").read_text())}
    paths = list((run_dir / "records").glob("*.json"))
    if len(roster) != 60 or len(paths) != 240:
        raise ValueError("Freeze requires all 240 completed generations")
    records = [json.loads(p.read_text()) for p in paths]
    by_key = {(r["item_id"], r["slot"]): r for r in records}
    if set(by_key) != {(i, s) for i in roster for s in SLOTS} or len(by_key) != 240:
        raise ValueError("Incorrect official slot keys")
    ordered = []
    for item, metadata in roster.items():
        calls = [by_key[item, s] for s in SLOTS]
        if (
            len({c["rgb_sha256"] for c in calls[:3]}) != 1
            or len({c["formatted_prompt_sha256"] for c in calls}) != 1
        ):
            raise ValueError("Input changed across repeated calls")
        for r in calls:
            parsed = r["final"].strip()
            valid = parsed in metadata["options"]
            if (r["status"] == "ok") != valid or r["count"] != (
                int(parsed) if valid else None
            ):
                raise ValueError("Final text and accepted count disagree")
            indices = r["source_indices"]
            if (
                len(indices) != 16
                or sorted(set(indices)) != indices
                or len(r["image_dimensions"]) != 16
                or any(max(x) > 224 for x in r["image_dimensions"])
            ):
                raise ValueError("Invalid frame budget or geometry")
            ordered.append({k: r[k] for k in FIELDS})
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in ordered)
    pred = out / "predictions.jsonl"
    if pred.exists() and pred.read_text() != data:
        raise ValueError(
            "A different prediction ledger is already frozen at this location"
        )
    pred.write_text(data)
    settings = json.loads((run_dir / "runtime.json").read_text())
    settings["completed_generations"] = len(records)
    settings["acquisition_dates_utc"] = sorted({r["started_utc"][:10] for r in records})
    (out / "runtime.json").write_text(json.dumps(settings, indent=2) + "\n")
    lock = {
        "scheduled_slots": 240,
        "terminal_slots": 240,
        "unique_items": 60,
        "predictions_sha256": sha(pred),
        "runtime_sha256": sha(out / "runtime.json"),
        "cohort_sha256": sha(HERE / "cohort.json"),
        "study_design_sha256": sha(HERE / "design_hashes.json"),
    }
    (out / "prediction_lock.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    freeze(args.run_dir, args.out)
