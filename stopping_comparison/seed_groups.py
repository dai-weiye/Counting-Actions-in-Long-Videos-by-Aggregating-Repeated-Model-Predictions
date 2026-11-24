"""Replay stopping policies across six frame-sampling groups for the same 32 videos."""

import argparse
import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

from compare import POLICIES, file_digest, load_json, policies, summarize


EXPERIMENTS = {
    "deepseek": ("seed_replication/main_design.json", "seed_replication/main_terminal_counts.jsonl",
                 "seed_replication/main_results.json"),
    "qwen": ("open_weight_seed/design.json", "open_weight_seed/predictions.jsonl", "open_weight_seed/results.json"),
}


def read_groups(package, design_file, counts_file, panel, options):
    design = load_json(package / design_file)
    ids = [Path(name).stem for name in design["filenames"]]
    if len(ids) != 32 or set(ids) != panel:
        raise ValueError(f"Incorrect video panel in {design_file}")
    expected = {(item, g, k) for item in ids for g in range(1, 7) for k in range(1, 5)}
    key = lambda row: (row["item_id"], row["group"], row["call"])
    scheduled = {key(s): s["frame_indices"] for s in design["slots"]}
    terminals = [json.loads(line) for line in (package / counts_file).read_text().splitlines()]
    if len(design["slots"]) != len(expected) or set(scheduled) != expected:
        raise ValueError(f"Missing or duplicated acquisition plan in {design_file}")
    if len(terminals) != len(expected) or {key(r) for r in terminals} != expected:
        raise ValueError(f"Missing or duplicated terminal record in {counts_file}")
    for row in terminals:
        value = row["count"]
        if row["status"] == "ok":
            if type(value) is not int or value not in options[row["item_id"]]:
                raise ValueError(f"Invalid numeric report at {key(row)}")
        elif value is not None:
            raise ValueError(f"Unusable terminal record contains a count at {key(row)}")
        if "frame_indices" in row and row["frame_indices"] != scheduled[key(row)]:
            raise ValueError(f"Frame indices differ from the plan at {key(row)}")
    return {key(r): r["count"] for r in terminals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package
    spec = spec_from_file_location("original_exact_stopping", package / "exact_stopping/replay.py")
    rule = module_from_spec(spec)
    spec.loader.exec_module(rule)
    with (package / "data/action_count.csv").open() as stream:
        gold = {r["item_id"]: int(r["gold"]) for r in csv.DictReader(stream)}
    options = {Path(r["video"]).stem: tuple(map(int, r["candidates"])) for r in load_json(package / "fourth_call/options.json")}
    panel = {Path(r["video"]).stem for r in load_json(package / "open_weight_check/cohort.json")}
    summaries, all_rows = {}, []
    sources = ["exact_stopping/replay.py", "data/action_count.csv", "fourth_call/options.json", "open_weight_check/cohort.json"]
    for model, (design_file, counts_file, old_report) in EXPERIMENTS.items():
        reports = read_groups(package, design_file, counts_file, panel, options)
        sources.extend([design_file, counts_file, old_report])
        original = load_json(package / old_report)["all_planned_by_group"]
        summaries[model] = {}
        for domain in ("fixed", "offered"):
            grouped, rows = {}, []
            for g in range(1, 7):
                group_rows = []
                for item in sorted(panel):
                    choices = options[item]
                    candidates = range(1, 6) if domain == "fixed" else choices
                    values = tuple(reports[item, g, k] for k in range(1, 5))
                    reference = rule.aggregate(values, candidates)
                    for name, used, predicted in policies(rule, values, choices, candidates):
                        if name in ("three_agree", "exact") and predicted != reference:
                            raise AssertionError((model, domain, g, item, name))
                        group_rows.append({"model": model, "domain": domain, "group": g, "item_id": item,
                                           "gold": gold[item], "policy": name, "calls": used,
                                           "valid_reports_used": sum(v is not None for v in values[:used]),
                                           "prediction": predicted, "four_slot_count": reference,
                                           "matches_four": predicted == reference, "correct": predicted == gold[item]})
                grouped[str(g)] = {p: summarize([r for r in group_rows if r["policy"] == p]) for p in POLICIES}
                if domain == "fixed":
                    baseline = grouped[str(g)]["fixed_4"]
                    archived = original[g - 1]
                    if (baseline["correct"], baseline["answered"], baseline["valid_reports_used"]) != (
                            archived["fallback_correct"], archived["at_least_one_valid"], archived["valid_reports"]):
                        raise ValueError(f"Archived four-slot accounting differs for {model}, group {g}")
                rows.extend(group_rows)
            totals = {p: summarize([r for r in rows if r["policy"] == p]) for p in POLICIES}
            summaries[model][domain] = {"by_group": grouped, "across_repeated_groups": totals}
            all_rows.extend(rows)
    result = {"unique_videos_per_model": len(panel), "groups_per_video": 6, "video_groups_per_model": 6 * len(panel),
              "source_sha256": {s: file_digest(package / s) for s in sources}, "by_model": summaries}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seed_groups.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / "seed_items.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
