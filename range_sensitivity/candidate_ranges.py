"""Post hoc candidate-range sensitivity on the common complete Qwen seed panel."""

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(reports, candidates):
    # Minimize twice the total deviation minus the number of exact matches.
    # The second key selects the smaller count whenever two totals coincide.
    def key(count):
        return sum(2 * abs(r - count) - (r == count) for r in reports), count

    return min(candidates, key=key)


def compare(package):
    directory = package / "open_weight_seed"
    inputs = {
        "reports": directory / "predictions.jsonl",
        "design": directory / "design.json",
        "options": package / "fourth_call" / "options.json",
        "labels": package / "data" / "action_count.csv",
    }
    hashes = {name: digest(path) for name, path in inputs.items()}
    if hashes["reports"] != load(directory / "runtime.json")["predictions_sha256"]:
        raise ValueError("Released prediction checksum differs from the run record")
    if hashes["design"] != load(directory / "design_lock.json")["sha256"]["design.json"]:
        raise ValueError("The locked seed design differs from the supplied design")

    videos = [Path(v).stem for v in load(inputs["design"])["filenames"]]
    groups, calls = range(1, 7), range(1, 5)
    records = [json.loads(line) for line in inputs["reports"].read_text().splitlines()]
    slots = {(r["item_id"], r["group"], r["call"]): r for r in records}
    if len(set(videos)) != 32 or len(records) != 768 or set(slots) != set(product(videos, groups, calls)):
        raise ValueError("Expected exactly 24 separate reports for each of 32 videos")
    options = {
        Path(r["video"]).stem: tuple(map(int, r["candidates"]))
        for r in load(inputs["options"])
    }
    with inputs["labels"].open(encoding="utf-8", newline="") as source:
        gold = {r["item_id"]: int(r["gold"]) for r in csv.DictReader(source)}
    for (video, _, _), record in slots.items():
        count = record["count"]
        if record["status"] == "ok":
            if type(count) is not int or count not in options[video]:
                raise ValueError("Accepted response falls outside the offered options")
        elif count is not None:
            raise ValueError("Unusable response carries a count")

    common = [v for v in videos if all(slots[v, g, k]["status"] == "ok" for g, k in product(groups, calls))]
    n = len(common)
    if n == 0:
        raise ValueError("No video has all 24 valid reports")
    summaries = []
    outside = 0
    for group in groups:
        fixed_correct = offered_correct = raw_correct = clipped_correct = 0
        for video in common:
            values = [slots[video, group, k]["count"] for k in calls]
            answer = gold[video]
            fixed_correct += estimate(values, range(1, 6)) == answer
            offered_correct += estimate(values, options[video]) == answer
            raw_correct += sum(v == answer for v in values)
            clipped_correct += sum(min(5, max(1, v)) == answer for v in values)
            outside += sum(v < 1 or v > 5 for v in values)
        summaries.append({
            "group": group,
            "fixed_four_correct": fixed_correct,
            "offered_four_correct": offered_correct,
            "raw_single_correct_out_of_4n": raw_correct,
            "clipped_single_correct_out_of_4n": clipped_correct,
            "fixed_minus_raw_pp": 100 * (4 * fixed_correct - raw_correct) / (4 * n),
            "fixed_minus_clipped_pp": 100 * (4 * fixed_correct - clipped_correct) / (4 * n),
            "offered_minus_raw_pp": 100 * (4 * offered_correct - raw_correct) / (4 * n),
        })
    return {
        "analysis": "Post hoc candidate-range sensitivity after completion of the six Qwen seed groups.",
        "contrast": "Four-report accuracy minus the mean accuracy of its four constituent single calls in percentage points.",
        "cohort": "Intersection of videos with 24 valid listed responses. Each group uses the same videos.",
        "planned_videos": len(videos),
        "common_videos": n,
        "constituent_single_reports_per_group": 4 * n,
        "common_reports_outside_fixed_range": outside,
        "groups": summaries,
        "mean_gain_pp": {k: mean(g[k] for g in summaries) for k in (
            "fixed_minus_raw_pp", "fixed_minus_clipped_pp", "offered_minus_raw_pp"
        )},
        "input_sha256": hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="Root of the anonymous reproducibility package")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args.package), indent=2, allow_nan=False) + "\n", encoding="utf-8")
