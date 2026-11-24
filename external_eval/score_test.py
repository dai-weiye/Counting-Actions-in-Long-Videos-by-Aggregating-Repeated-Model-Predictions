"""Offline scoring of frozen MLVU Test predictions using the development scorer."""

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")


def scores(items, records, methods):
    by_key = {(r["id"], r["slot"]): r for r in records}
    if len(by_key) != len(records) or set(by_key) != {
        (r["id"], slot) for r in items for slot in SLOTS
    }:
        raise ValueError(
            "Expect exactly one terminal record in all four slots for every planned question"
        )
    if len(items) != len({r["id"] for r in items}) or len(items) != 60:
        raise ValueError("Expected the full 60-question test split")
    both_truth, first, prefix, offered_four, outputs = [], [], [], [], {}
    rows, statuses = [], {s: Counter() for s in SLOTS}
    triples = []
    for item in items:
        key, gold = item["id"], item["gold"]
        assert gold in item["options"] and len(item["options"]) == 6
        reports = []
        for slot in SLOTS:
            obs = by_key[key, slot]
            statuses[slot][obs["status"]] += 1
            v = obs["count"]
            if obs["status"] == "ok":
                if type(v) != int or v not in item["options"]:
                    raise ValueError(f"Invalid count for {key}/{slot}")
            elif v is not None:
                raise ValueError(f"Unusable response has a count at {key}/{slot}")
            reports.append(v)
        if all(v is not None for v in reports[:3]):
            triples.append(len(set(reports[:3])) > 1)
        available = [v for v in reports if v is not None]
        complete = len(available) == 4
        pred = methods.predictions(available) if available else None
        offered_pred = (
            min(
                item["options"],
                key=lambda c: (
                    -sum(-2 * abs(v - c) + int(v == c) for v in available),
                    c,
                ),
            )
            if available
            else None
        )
        if complete:
            offered_four.append(offered_pred)
            both_truth.append(gold)
            first.append(reports[0])
            prefix.append(methods.predictions(reports[:3])["exact_lower"])
            for rule, v in pred.items():
                outputs.setdefault(rule, []).append(v)
        rows.append(
            {
                "id": key,
                "gold": gold,
                "U1": reports[0],
                "four_complete": pred["exact_lower"] if complete else None,
                "offered_options_four_complete": offered_pred if complete else None,
                "offered_options_available_fallback": offered_pred,
                "available_report_fallback": pred["exact_lower"] if pred else None,
                "duration_tree_trained_on_dev": item["dev_duration_prediction"],
            }
        )
    doc = {
        "n": len(items),
        "terminal_slots": len(records),
        "reference_count_histogram": dict(
            sorted(Counter(row["gold"] for row in rows).items())
        ),
        "references_outside_1_to_5": sum(not 1 <= row["gold"] <= 5 for row in rows),
        "complete_case_references_outside_1_to_5": sum(
            not 1 <= gold <= 5 for gold in both_truth
        ),
        "status_per_slot": statuses,
        "identical_uniform_input": {
            "three_valid": len(triples),
            "different_numeric_reports": sum(triples),
        },
        "four_valid": len(both_truth),
        "matched_MAMC_minus_U1": {
            k: methods.paired(v, first, both_truth) for k, v in outputs.items()
        },
        "matched_offered_options_four_minus_U1": methods.paired(
            offered_four, first, both_truth
        ),
        "matched_four_minus_range_matched_U1": methods.paired(
            outputs.get("exact_lower", []),
            [min(5, max(1, v)) for v in first],
            both_truth,
        ),
        "matched_four_minus_three": methods.paired(
            outputs.get("exact_lower", []), prefix, both_truth
        ),
        "matched_four_minus_vote": methods.paired(
            outputs.get("exact_lower", []), outputs.get("majority", []), both_truth
        ),
        "all_planned": {},
    }
    for k in (
        "U1",
        "four_complete",
        "available_report_fallback",
        "offered_options_four_complete",
        "offered_options_available_fallback",
        "duration_tree_trained_on_dev",
    ):
        observed = [row for row in rows if row[k] is not None]
        correct = sum(row[k] == row["gold"] for row in observed)
        doc["all_planned"][k] = {
            "answered": len(observed),
            "correct": correct,
            "n": len(rows),
            "correct_per_planned_pct": 100 * correct / len(rows),
            "mae_given_answer": sum(abs(row[k] - row["gold"]) for row in observed)
            / len(observed)
            if observed
            else None,
        }
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "items.json")
    parser.add_argument("--calls", type=Path, default=ROOT / "calls.jsonl")
    parser.add_argument("--out", type=Path)
    package = (
        ROOT.parent
        if (ROOT.parent / "fourth_call").is_dir()
        else ROOT.parents[1] / "paper/ICME2027_REPRODUCIBILITY"
    )
    parser.add_argument("--protocol-root", type=Path, default=package)
    opts = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "mamc_score", opts.protocol_root / "fourth_call/analyze.py"
    )
    methods = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(methods)
    rows = json.loads(opts.items.read_text())
    calls = [json.loads(l) for l in opts.calls.read_text().splitlines()]
    from sklearn.tree import DecisionTreeClassifier

    with (opts.protocol_root / "data/action_count.csv").open() as stream:
        training = sorted(csv.DictReader(stream), key=lambda r: r["item_id"] + ".mp4")
    assert len(training) == 206
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        [[float(r["duration_s"])] for r in training], [int(r["gold"]) for r in training]
    )
    assert tree.predict([[r["duration"]] for r in rows]).tolist() == [
        r["dev_duration_prediction"] for r in rows
    ]
    results = json.dumps(scores(rows, calls, methods), indent=2, allow_nan=False) + "\n"
    if opts.out:
        opts.out.write_text(results)
    else:
        print(results, end="")


if __name__ == "__main__":
    main()
