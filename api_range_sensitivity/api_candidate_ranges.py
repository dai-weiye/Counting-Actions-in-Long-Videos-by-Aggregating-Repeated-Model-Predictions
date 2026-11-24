"""Post hoc candidate-range comparisons for the six API frame-sampling groups."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path


GROUPS = range(1, 7)
CALLS = range(1, 5)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(reports, candidates):
    if not reports:
        return None

    def rank(c):
        return sum(2 * abs(v - c) - int(v == c) for v in reports), c

    return min(candidates, key=rank)


def summarize(counts, n):
    gains = {}
    for method in ("strict", "available"):
        fixed = 4 * counts[method + "_fixed_correct"]
        offered = 4 * counts[method + "_offered_correct"]
        gains[method] = {
            "fixed_minus_raw_pp": 100 * (fixed - counts["raw_single_correct"]) / (4 * n),
            "fixed_minus_clipped_pp": 100 * (fixed - counts["clipped_single_correct"]) / (4 * n),
            "offered_minus_raw_pp": 100 * (offered - counts["raw_single_correct"]) / (4 * n),
        }
    return {"video_groups": n, "single_slots": 4 * n, **counts, "gain_pp": gains}


def cohort(videos, slots, options, gold):
    groups = []
    total = Counter()
    for group in GROUPS:
        counts = Counter()
        for video in videos:
            reports = [slots[video, group, k]["count"] for k in CALLS]
            valid = [v for v in reports if v is not None]
            label = gold[video]
            fixed = estimate(valid, range(1, 6))
            offered = estimate(valid, options[video])
            counts["usable_single_slots"] += len(valid)
            counts["raw_single_correct"] += sum(v == label for v in valid)
            counts["clipped_single_correct"] += sum(min(5, max(1, v)) == label for v in valid)
            counts["strict_answered"] += len(valid) == 4
            counts["available_answered"] += bool(valid)
            for name, value in (("fixed", fixed), ("offered", offered)):
                counts["strict_" + name + "_correct"] += len(valid) == 4 and value == label
                counts["available_" + name + "_correct"] += value == label
        groups.append({"group": group, **summarize(counts, len(videos))})
        total.update(counts)
    return {"videos": len(videos), "groups": groups, "overall": summarize(total, len(videos) * len(GROUPS))}


def analyze(package):
    seed = package / "seed_replication"
    inputs = {
        "reports": seed / "main_terminal_counts.jsonl",
        "design": seed / "main_design.json",
        "options": package / "fourth_call" / "options.json",
        "labels": package / "data" / "action_count.csv",
    }
    hashes = {key: sha256(path) for key, path in inputs.items()}
    design = load(inputs["design"])
    if hashes["design"] != load(seed / "acquisition.json")["main"]["input_design_sha256"]:
        raise ValueError("Seed panel differs from its recorded design")
    ids = [Path(name).stem for name in design["filenames"]]
    options = {Path(v["video"]).stem: tuple(map(int, v["candidates"])) for v in load(inputs["options"])}
    with inputs["labels"].open(encoding="utf-8", newline="") as source:
        gold = {r["item_id"]: int(r["gold"]) for r in csv.DictReader(source)}
    records = [json.loads(line) for line in inputs["reports"].read_text(encoding="utf-8").splitlines()]
    slots = {(r["item_id"], r["group"], r["call"]): r for r in records}
    planned = {(r["item_id"], r["group"], r["call"]) for r in design["slots"]}
    if len(ids) != 32 or len(records) != 768 or set(slots) != planned or planned != set(product(ids, GROUPS, CALLS)):
        raise ValueError("Expected 32 videos, six groups and four terminal slots for every video and group")
    for (video, _, _), r in slots.items():
        if r["status"] == "ok":
            if type(r["count"]) is not int or r["count"] not in options[video]:
                raise ValueError("Accepted response is not an offered integer")
        elif r["count"] is not None:
            raise ValueError("Missing or invalid response carries a numeric count")
    common = [v for v in ids if all(slots[v, g, k]["status"] == "ok" for g, k in product(GROUPS, CALLS))]
    if not common:
        raise ValueError("No video has all 24 usable reports")
    return {
        "analysis": "Post hoc candidate-range sensitivity after completion of all six API seed groups.",
        "contrast": "Four-report accuracy minus the mean accuracy of its four constituent single calls, in percentage points.",
        "rules": {
            "strict": "Requires four usable reports; an unanswered video-group counts as incorrect.",
            "available": "Scores all usable reports; an unanswered video-group counts as incorrect.",
            "fixed": "Score integers 1 through 5, breaking ties toward the smaller count.",
            "offered": "Score the video's listed numerical options, breaking ties toward the smaller count.",
            "clipped_single": "Clip each usable single response to 1 through 5; an unanswered slot counts as incorrect.",
        },
        "common_complete": cohort(common, slots, options, gold),
        "all_planned": cohort(ids, slots, options, gold),
        "input_sha256": hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Root of the anonymous reproducibility package")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.package), indent=2, allow_nan=False) + "\n", encoding="utf-8")
