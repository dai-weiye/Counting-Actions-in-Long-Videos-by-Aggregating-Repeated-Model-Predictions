"""Compare stopping times question by question on the two DeepSeek evaluation cohorts."""

import argparse
import csv
import json
from pathlib import Path


COHORTS = {"dev_held_out": "development_held_out", "test": "official_test_all"}
POLICIES = ("exact", "two_agree", "three_agree")


def read_items(folder, domain):
    items = {split: {} for split in COHORTS}
    with (folder / "items.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split, name = row["split"], row["policy"]
            if split not in items:
                continue
            group = items[split].setdefault(row["item_id"], {})
            if name == "fixed_1":
                # Every usable report is an offered choice. The fixed range clips it to 1..5.
                first = "first_raw" if row["domain"] == "offered" else "first_clipped"
                group[first] = row
            if row["domain"] == domain:
                if name in group:
                    raise ValueError(f"Duplicate {name} for {split}/{row['item_id']}")
                group[name] = row
    return items


def summarize_policy(items, name):
    stops = [0] * 4
    disagreements = [0] * 4
    answered = correct = 0
    for rows in items.values():
        if name == "first_usable":
            # At its first usable prefix the cumulative scorer has exactly one report.
            prefixes = [rows[f"fixed_{k}"] for k in range(1, 5)]
            row = next((r for r in prefixes if int(r["valid_reports_used"]) > 0), prefixes[-1])
        else:
            row = rows[name]
        slot = int(row["calls"])
        if slot not in range(1, 5):
            raise ValueError(f"Invalid stopping slot for {row['item_id']}/{name}")
        stops[slot - 1] += 1
        prediction = row["prediction"]
        # Empty fields represent unanswered items and compare equal to each other.
        disagreements[slot - 1] += prediction != rows["fixed_4"]["prediction"]
        answered += prediction != ""
        correct += prediction == rows["fixed_4"]["gold"]
    result = {
        "calls": sum(k * count for k, count in enumerate(stops, start=1)),
        "matches_four": len(items) - sum(disagreements),
        "answered": answered,
        "correct": correct,
    }
    if name in (*POLICIES, "first_usable"):
        result.update(stops_by_slot=stops, disagreements_with_four=sum(disagreements),
                      disagreements_by_stop_slot=disagreements)
    return result


def summarize(items, published):
    names = (*POLICIES, "first_usable", "first_raw", "first_clipped")
    policies = {name: summarize_policy(items, name) for name in names}
    for name in POLICIES:
        reference = published[name]
        if len(items) != reference["n"] or any(
            policies[name][key] != reference[key] for key in ("calls", "matches_four", "answered", "correct")
        ):
            raise ValueError(f"Totals differ from published results for {name}")
    comparisons = {}
    for name in ("two_agree", "three_agree"):
        differences = [int(rows["exact"]["calls"]) - int(rows[name]["calls"]) for rows in items.values()]
        comparisons[name] = {
            "earlier": sum(d < 0 for d in differences),
            "same": differences.count(0),
            "later": sum(d > 0 for d in differences),
        }
    return {"n": len(items), "exact_stop_relative_to": comparisons, "policies": policies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path, help="extracted release directory")
    parser.add_argument("--root", type=Path, help="alternative to the positional release directory")
    parser.add_argument("--domain", choices=("fixed", "offered"), default="fixed")
    parser.add_argument("--out", type=Path, help="write JSON here (default: standard output)")
    args = parser.parse_args()
    if args.package and args.root:
        parser.error("supply the release directory once")
    folder = (args.root or args.package or Path(".")) / "stopping_comparison"
    published = json.loads((folder / "results.json").read_text(encoding="utf-8"))["by_domain"][args.domain]
    items = read_items(folder, args.domain)
    result = {"domain": args.domain, "slot_order": ["U1", "U2", "U3", "C4"]}
    for split, cohort in COHORTS.items():
        result[cohort] = summarize(items[split], published[cohort])
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
