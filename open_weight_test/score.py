"""Score the 240 frozen Qwen test outputs and write only aggregate results."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")
ANNOTATION_SHA256 = "b2fffd0cddbdf0ed54592db33e46979b20082bb00ee3db828e820f5080d87262"


def module_from(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(args):
    pred = args.predictions
    lock = json.loads(args.lock.read_text())
    if (
        lock["scheduled_slots"] != 240
        or lock["terminal_slots"] != 240
        or lock["predictions_sha256"] != digest(pred)
    ):
        raise ValueError(
            "All model outputs must be frozen before reading reference labels"
        )
    if (
        digest(HERE / "cohort.json") != lock["cohort_sha256"]
        or digest(HERE / "design_hashes.json") != lock["study_design_sha256"]
    ):
        raise ValueError("The frozen study design or cohort changed")
    roster = {q["id"]: q for q in json.loads((HERE / "cohort.json").read_text())}
    records = [json.loads(line) for line in pred.read_text().splitlines()]
    ledger = {(r["item_id"], r["slot"]): r for r in records}
    if (
        len(records) != 240
        or len(roster) != 60
        or set(ledger) != {(i, s) for i in roster for s in SLOTS}
    ):
        raise ValueError("Unrecognized test questions or incomplete rows")
    if args.annotations:
        import pyarrow.parquet as pq

        if digest(args.annotations) != ANNOTATION_SHA256:
            raise ValueError("Annotation file checksum mismatch")
        letters = {
            r["question_id"]: r["answer"].strip()
            for r in pq.read_table(
                args.annotations, columns=["question_id", "task_type", "answer"]
            ).to_pylist()
            if r["task_type"] == "count"
        }
        if set(letters) != set(roster) or any(
            x not in tuple("ABCDEF") for x in letters.values()
        ):
            raise ValueError("Invalid official test labels")
        gold = {
            i: int(r["options"][ord(letters[i]) - ord("A")]) for i, r in roster.items()
        }
        if args.export_labels:
            args.export_labels.write_text(json.dumps(gold, indent=2) + "\n")
    else:
        gold = json.loads(args.labels.read_text())
        if set(gold) != set(roster) or any(
            type(gold[i]) is not int or str(gold[i]) not in r["options"]
            for i, r in roster.items()
        ):
            raise ValueError(
                "Released label manifest does not match the frozen test roster"
            )
    counts = module_from(
        "qwen_count_aggregation", HERE.parent / "open_weight_check/score.py"
    )
    certificate = module_from(
        "qwen_test_stopping", HERE.parent / "exact_stopping/replay.py"
    )
    duration = json.loads((HERE / "duration_predictions.json").read_text())
    totals = Counter()
    correct = Counter()
    complete_gold = []
    singles = []
    clipped = []
    lower = []
    offered = []
    stop_rows = []
    statuses = {s: Counter() for s in SLOTS}
    levels = Counter()
    complete_by_rule = Counter()
    for item, q in roster.items():
        options = [int(v) for v in q["options"]]
        y = gold[item]
        levels[y] += 1
        reports = [ledger[item, s]["count"] for s in SLOTS]
        for slot in SLOTS:
            statuses[slot][ledger[item, slot]["status"]] += 1
        ok = [v for v in reports if v is not None]
        u = reports[0]
        c4 = reports[-1]
        supported = min(5, max(1, u)) if u is not None else None
        score_value = lambda c: sum(-2 * abs(v - c) + int(v == c) for v in ok)
        agg = counts.aggregate(ok) if len(ok) == 4 else None
        offered_pred = max(sorted(options), key=score_value) if ok else None
        if None not in reports[:3]:
            totals["three_uniform_usable"] += 1
            totals["three_uniform_disagree"] += len(set(reports[:3])) > 1
        if u is not None and c4 is not None:
            totals["both_u1_c4_usable"] += 1
            totals["u1_c4_disagree"] += u != c4
        if agg is not None:
            complete_gold.append(y)
            singles.append(u)
            clipped.append(supported)
            lower.append(agg)
            offered.append(offered_pred)
            totals["four_equals_clipped_u1"] += agg == supported
            byrule = {
                "single": u,
                "clipped_median_three": min(5, max(1, sorted(reports[:3])[1])),
                "vote_three": Counter(reports[:3]).most_common(1)[0][0],
                "vote_four": Counter(reports).most_common(1)[0][0],
                "lower_four": agg,
                "upper_four": max(reversed(range(1, 6)), key=score_value),
                "offered_four": offered_pred,
            }
            for k, v in byrule.items():
                complete_by_rule[k] += v == y
        decisions = {
            "single": u,
            "range_matched_single": supported,
            "four_required": agg,
            "available_report_fallback": counts.aggregate(ok),
            "duration_tree": duration[item],
        }
        for k, v in decisions.items():
            totals[k + "_answered"] += v is not None
            correct[k] += v == y
        stop_rows.append(certificate.replay(item, options, reports))
    return {
        "planned_questions": 60,
        "terminal_slots": 240,
        "status_by_slot": statuses,
        "complete_pairs": len(complete_gold),
        "paired_four_vs_raw_first": counts.contrast(lower, singles, complete_gold),
        "paired_four_vs_clipped_first": counts.contrast(lower, clipped, complete_gold),
        "paired_offered_four_vs_raw_first": counts.contrast(
            offered, singles, complete_gold
        ),
        "counts": totals,
        "correct_all_planned": correct,
        "complete_cohort_correct": complete_by_rule,
        "reference_count_histogram": dict(sorted(levels.items())),
        "reference_outside_candidates": sum(
            n for c, n in levels.items() if not 1 <= c <= 5
        ),
        "exact_stopping": certificate.summarize(stop_rows),
        "input_hashes": {
            "predictions": digest(pred),
            "annotations": ANNOTATION_SHA256,
            "duration_predictions": digest(HERE / "duration_predictions.json"),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=HERE / "predictions.jsonl")
    p.add_argument("--lock", type=Path, default=HERE / "prediction_lock.json")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--annotations", type=Path)
    source.add_argument("--labels", type=Path, default=HERE / "official_counts.json")
    p.add_argument(
        "--export-labels",
        type=Path,
        help="Export the 60 public numeric answers when scoring the pinned parquet",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(score(args), indent=2, allow_nan=False) + "\n")
