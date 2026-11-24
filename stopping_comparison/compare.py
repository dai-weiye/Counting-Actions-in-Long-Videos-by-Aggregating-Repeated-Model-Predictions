"""Compare sequential counting rules on the same archived four reports for each question."""

import argparse
import csv
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np


SOURCE = {
    "action_count.csv": "data/action_count.csv",
    "options.json": "fourth_call/options.json",
    "official_test_cohort": "external_eval/items.json",
    "official_test_calls": "external_eval/calls.jsonl",
    "open_weight_official_test_cohort": "open_weight_test/cohort.json",
    "open_weight_official_test_calls": "open_weight_test/predictions.jsonl",
    "open_weight_held_out_calls": "open_weight_check/predictions.jsonl",
}
POLICIES = ("fixed_1", "fixed_2", "fixed_3", "fixed_4", "three_agree", "two_agree", "exact")
GROUPS = {
    "development_all": {"dev_fit", "dev_held_out"},
    "development_held_out": {"dev_held_out"},
    "official_test_all": {"test"},
    "open_weight_official_test_all": {"open_weight_test"},
    "open_weight_held_out": {"open_weight_held_out"},
}


def load_json(path):
    return json.loads(path.read_text())


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_items(package, slots):
    source = {key: package / rel for key, rel in SOURCE.items()}
    with source["action_count.csv"].open() as stream:
        dev = sorted(csv.DictReader(stream), key=lambda r: r["item_id"] + ".mp4")
    options = {Path(r["video"]).stem: tuple(map(int, r["candidates"])) for r in load_json(source["options.json"])}
    dev_labels = {r["item_id"]: int(r["gold"]) for r in dev}
    if len(dev) != 206 or set(dev_labels) != set(options) or len(dev_labels) != 206:
        raise ValueError("Development IDs and options disagree")
    holdout = set(np.random.default_rng(0).permutation(len(dev))[103:])
    data = []
    for i, row in enumerate(dev):
        item = row["item_id"]
        values = tuple(int(row[k]) for k in ("uniform", "random", "recency", "change"))
        data.append(("dev_held_out" if i in holdout else "dev_fit", item, options[item], dev_labels[item], values))

    api = load_json(source["official_test_cohort"])
    api_labels = {q["id"]: int(q["gold"]) for q in api}
    qwen = load_json(source["open_weight_official_test_cohort"])
    qwen_labels = load_json(package / "open_weight_test/official_counts.json")
    if len(api) != 60 or len(qwen) != 60 or len(api_labels) != 60 or qwen_labels != api_labels:
        raise ValueError("Official API and Qwen test labels disagree")
    for split, src, id_key, cohort in (
        ("test", "official_test_calls", "id", api),
        ("open_weight_test", "open_weight_official_test_calls", "item_id", qwen),
        ("open_weight_held_out", "open_weight_held_out_calls", "item_id", None),
    ):
        records = [json.loads(line) for line in source[src].read_text().splitlines()]
        lookup = {(r[id_key], r["slot"]): r for r in records}
        if cohort is None:
            ids = sorted({r[id_key] for r in records})
            if len(ids) != 32 or not set(ids) <= {dev[i]["item_id"] for i in holdout}:
                raise ValueError("Invalid 32-item held-out panel")
            questions = [(i, options[i], dev_labels[i]) for i in ids]
        else:
            questions = sorted((q["id"], tuple(map(int, q["options"])), api_labels[q["id"]]) for q in cohort)
        if len(lookup) != len(records) or set(lookup) != {(i, s) for i, _, _ in questions for s in slots}:
            raise ValueError(f"Missing or repeated slot in {split}")
        for i, opts, gold in questions:
            ordered = [lookup[i, s] for s in slots]
            if any((r["status"] == "ok") != (r["count"] is not None) for r in ordered):
                raise ValueError(f"Inconsistent terminal status for {split} {i}")
            data.append((split, i, opts, gold, tuple(r["count"] for r in ordered)))
    return data


def policies(rule, reports, options, candidates):
    for k in range(1, 5):
        yield f"fixed_{k}", k, rule.aggregate(reports[:k], candidates)
    for name, k in (("three_agree", 3), ("two_agree", 2)):
        prefix = reports[:k]
        used = k if prefix[0] is not None and len(set(prefix)) == 1 else 4
        yield name, used, rule.aggregate(reports[:used], candidates)
    decision, used = rule.count_sequential(options, iter(reports), candidates)
    yield "exact", used, decision


def summarize(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    same = sum(r["matches_four"] for r in rows)
    return {"n": n, "calls": sum(r["calls"] for r in rows),
            "valid_reports_used": sum(r["valid_reports_used"] for r in rows),
            "matches_four": same, "differs_from_four": n - same,
            "answered": sum(r["prediction"] is not None for r in rows), "correct": correct,
            "accuracy_pct_all_items": 100 * correct / n}


def check_original(package, rule, domain, records, source_hashes):
    suffix = "_offered" if domain == "offered" else ""
    frozen = package / "exact_stopping"
    # Match every column of every original item and every field in the cohort summaries.
    with (frozen / f"items{suffix}.csv").open() as stream:
        expected = list(csv.DictReader(stream))
    encoded = [{k: "" if v is None else str(v) for k, v in r.items()} for r in records]
    if expected != encoded:
        raise ValueError(f"{domain} exact decisions or call counts differ from the frozen item table")
    result = {}
    for group, splits in GROUPS.items():
        subset = [r for r in records if r["split"] in splits]
        result[group] = rule.summarize(subset)
        if group in ("official_test_all", "open_weight_official_test_all"):
            result[group.removesuffix("_all") + "_four_usable"] = rule.summarize([r for r in subset if r["four_usable"]])
    result["input_sha256"] = source_hashes
    if result != load_json(frozen / f"results{suffix}.json"):
        raise ValueError(f"{domain} exact summaries differ from the frozen results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="extracted anonymous reproduction package")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    package = args.package
    spec = spec_from_file_location("original_exact_stopping", package / "exact_stopping/replay.py")
    rule = module_from_spec(spec)
    spec.loader.exec_module(rule)
    source_hashes = {key: file_digest(package / rel) for key, rel in SOURCE.items()}
    inputs = load_items(package, rule.SLOTS)
    all_rows, summaries = [], {}
    for domain in ("fixed", "offered"):
        records, exact_records = [], []
        for split, item, options, gold, reports in inputs:
            if gold not in options or any(v is not None and (type(v) is not int or v not in options) for v in reports):
                raise ValueError(f"Unexpected count at {split} {item}")
            candidates = range(1, 6) if domain == "fixed" else options
            reference = rule.aggregate(reports, candidates)
            for name, used, predicted in policies(rule, reports, options, candidates):
                match = predicted == reference
                if name in ("three_agree", "exact") and not match:
                    raise AssertionError(f"Safe rule {name} disagrees on {domain} {split} {item}")
                out = {"domain": domain, "split": split, "item_id": item, "gold": gold,
                       "policy": name, "calls": used, "valid_reports_used": sum(v is not None for v in reports[:used]),
                       "prediction": predicted, "four_slot_count": reference, "matches_four": match, "correct": predicted == gold}
                records.append(out)
                if name == "exact":
                    exact_records.append({"split": split, "item_id": item, "four_usable": all(v is not None for v in reports),
                                          "used_calls": used, "saved_calls": 4 - used, "early_count": predicted, "four_slot_count": reference})
        check_original(package, rule, domain, exact_records, source_hashes)
        all_rows.extend(records)
        summaries[domain] = {
            group: {name: summarize([r for r in records if r["split"] in splits and r["policy"] == name]) for name in POLICIES}
            for group, splits in GROUPS.items()}
    provenance = {rel: source_hashes[key] for key, rel in SOURCE.items()}
    for rel in ("open_weight_test/official_counts.json", "exact_stopping/replay.py", "exact_stopping/items.csv",
                "exact_stopping/items_offered.csv", "exact_stopping/results.json", "exact_stopping/results_offered.json"):
        provenance[rel] = file_digest(package / rel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps({"source_sha256": provenance, "by_domain": summaries}, indent=2) + "\n")
    with (args.out_dir / "items.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
