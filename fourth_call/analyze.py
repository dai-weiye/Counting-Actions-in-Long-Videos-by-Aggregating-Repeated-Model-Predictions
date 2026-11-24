"""Analyze interleaved fourth-call results using the locally frozen study plan."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
from math import comb
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "paper/ICME2027_SUBMIT/ICME2027_Anonymous_Supplement.zip"
ANNOTATIONS = ROOT / "data/raw/mlvu/MLVU/json/4_count.json"
ARMS = ("uniform", "change")
RULES = ("exact_lower", "exact_upper", "archived_float", "majority", "rounded_median")


def inputs(paths, archived_csv=None, annotations=ANNOTATIONS):
    if archived_csv is None:
        with ZipFile(PACKAGE) as archive:
            csv_bytes = archive.read("ICME2027_REPRODUCIBILITY/data/action_count.csv")
    else:
        csv_bytes = archived_csv.read_bytes()
    annotation_bytes = annotations.read_bytes()
    rows = sorted(csv.DictReader(io.StringIO(csv_bytes.decode())), key=lambda r: r["item_id"] + ".mp4")
    if len(rows) != 206 or len({r["item_id"] for r in rows}) != 206:
        raise ValueError("The original split requires exactly 206 unique benchmark items")
    options = {Path(r["video"]).stem: set(map(int, r["candidates"])) for r in json.loads(annotation_bytes)}
    if set(options) != {r["item_id"] for r in rows}:
        raise ValueError("Public annotation IDs and archived IDs disagree")
    if any(len(offered) != 4 for offered in options.values()):
        raise ValueError("Each public item must offer four distinct counts")
    if not all(int(r["gold"]) in options[r["item_id"]] for r in rows):
        raise ValueError("A reference answer is absent from its public options")
    for row in rows:
        row["offered_counts"] = sorted(options[row["item_id"]])
    fingerprints = {
        "archived_numeric_csv": sha256(csv_bytes).hexdigest(),
        "public_annotations": sha256(annotation_bytes).hexdigest(), "terminal_files": [],
    }
    terminal = {}
    for path in paths:
        data = path.read_bytes()
        fingerprints["terminal_files"].append({"name": path.name, "sha256": sha256(data).hexdigest()})
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item, arm, status = row["item_id"], row["arm"], row["status"]
            key = item, arm
            if item not in options or arm not in ARMS or key in terminal:
                raise ValueError(f"{path}:{line_number}: unknown or duplicate {key}")
            count = row.get("count")
            if not isinstance(status, str) or not status or status == "not_recorded":
                raise ValueError(f"{path}:{line_number}: invalid terminal status")
            if status == "ok" and (type(count) is not int or count not in options[item]):
                raise ValueError(f"{path}:{line_number}: expected an integer from the item's options")
            if status != "ok" and count is not None:
                raise ValueError(f"{path}:{line_number}: failed record contains a count")
            terminal[key] = status, count
    return rows, terminal, fingerprints


def predictions(values):
    exact = []
    floating = []
    for c in range(1, 6):
        exact.append(sum(-2 * abs(v - c) + int(v == c) for v in values))
        total = 0.0
        for v in values:
            # Preserve the original floating-point operation and accumulation order.
            total += -0.8 * max(c - v, 0) - 0.8 * max(v - c, 0) + 0.4 * (v == c)
        floating.append(total)
    tied = [c for c, s in enumerate(exact, 1) if s == max(exact)]
    return {"exact_lower": tied[0], "exact_upper": tied[-1],
            "archived_float": 1 + floating.index(max(floating)),
            "majority": Counter(values).most_common(1)[0][0],
            "rounded_median": int(np.rint(np.median(values)))}


def paired(first, reference, truth):
    """All three arrays refer to the same items. The contrast is first minus reference."""
    if not len(truth):
        return None
    first, reference, truth = map(np.asarray, (first, reference, truth))
    n = len(truth)
    a, b = first == truth, reference == truth
    wins = int(np.sum(a & ~b))
    losses = int(np.sum(b & ~a))
    discordant = wins + losses
    tail = sum(comb(discordant, k) for k in range(min(wins, losses) + 1))
    p = min(1.0, 2 * tail / (2 ** discordant))
    # Resampling the three paired-outcome categories equals resampling questions.
    draws = np.random.default_rng(0).multinomial(n, np.array([wins, losses, n-discordant])/n, 20_000)
    interval = np.percentile(100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5])
    return {"n": n, "first_correct": int(a.sum()), "reference_correct": int(b.sum()),
            "first_accuracy_pct": 100 * float(a.mean()), "reference_accuracy_pct": 100 * float(b.mean()),
            "first_mae": float(np.abs(first - truth).mean()), "reference_mae": float(np.abs(reference - truth).mean()),
            "wins": wins, "losses": losses, "delta_pp": 100 * (wins-losses)/n,
            "delta_pp_ci95": interval.tolist(), "exact_mcnemar_p": p}


def summarize(cohort, terminal):
    status_counts = {arm: Counter() for arm in ARMS}
    availability = Counter({"both_valid": 0, "uniform_only": 0, "change_only": 0, "neither_valid": 0})
    outputs = {arm: {rule: [] for rule in RULES} for arm in ARMS}
    available_counts = {arm: Counter() for arm in ARMS}
    missing = []
    matched_gold, prefix = [], []
    lower, upper = 0, 0
    option_lower, option_upper = 0, 0
    unresolved = Counter({"uniform": 0, "change": 0})
    for row in cohort:
        item, gold = row["item_id"], int(row["gold"])
        three = [int(row[key]) for key in ("uniform", "random", "recency")]
        scores, status = {}, {}
        for arm in ARMS:
            status[arm], count = terminal.get((item, arm), ("not_recorded", None))
            status_counts[arm][status[arm]] += 1
            if status[arm] == "ok":
                scores[arm] = predictions([*three, count])
                decision = scores[arm]["exact_lower"]
                available_counts[arm].update(n=1, correct=decision == gold, absolute_error=abs(decision - gold))
        # Missing correctness can range from zero to one on the planned denominator.
        accuracy_bounds = {
            arm: (int(scores[arm]["exact_lower"] == gold),) * 2 if arm in scores else (0, 1)
            for arm in ARMS
        }
        lower += accuracy_bounds["change"][0] - accuracy_bounds["uniform"][1]
        upper += accuracy_bounds["change"][1] - accuracy_bounds["uniform"][0]
        # An additional sensitivity assumes each missing final would be a listed option.
        option_bounds = accuracy_bounds.copy()
        if len(scores) != 2:
            possible = [int(predictions([*three, option])["exact_lower"] == gold)
                        for option in row["offered_counts"]]
            for arm in ARMS:
                if arm not in scores:
                    option_bounds[arm] = min(possible), max(possible)
                    unresolved[arm] += min(possible) != max(possible)
        option_lower += option_bounds["change"][0] - option_bounds["uniform"][1]
        option_upper += option_bounds["change"][1] - option_bounds["uniform"][0]
        if len(scores) != 2:
            available = next(iter(scores), None)
            coverage_group = f"{available}_only" if available else "neither_valid"
            availability[coverage_group] += 1
            missing.append({"item_id": item, **status})
            continue
        availability["both_valid"] += 1
        matched_gold.append(gold)
        prefix.append(predictions(three)["exact_lower"])
        for arm in ARMS:
            for rule, value in scores[arm].items():
                outputs[arm][rule].append(value)
    n = len(cohort)
    contrasts = {rule: paired(outputs["change"][rule], outputs["uniform"][rule], matched_gold) for rule in RULES}
    available_per_arm = {}
    for arm, counts in available_counts.items():
        n_valid = counts["n"]
        available_per_arm[arm] = {"n": n_valid, "correct": counts["correct"],
                                  "accuracy_pct": 100 * counts["correct"] / n_valid if n_valid else None,
                                  "mae": counts["absolute_error"] / n_valid if n_valid else None}
    return {
        "planned_n": n, "all_arm_records_present": all(v["not_recorded"] == 0 for v in status_counts.values()),
        "complete_pairs": len(matched_gold), "all_pairs_available": n == len(matched_gold),
        "status_by_arm": status_counts, "pair_availability": availability, "unpaired": missing,
        "planned_primary_delta_pp_bounds": [100 * lower/n, 100 * upper/n],
        "sensitivity_valid_option_completions": {
            "planned_delta_pp_bounds": [100 * option_lower/n, 100 * option_upper/n],
            "missing_slots_with_unresolved_correctness": unresolved},
        "available_primary_per_arm_separate_denominators": available_per_arm,
        "matched_change_minus_uniform": contrasts,
        "matched_primary_minus_three_uniform": {
            arm: paired(outputs[arm]["exact_lower"], prefix, matched_gold) for arm in ARMS},
        "matched_primary_minus_majority": {
            arm: paired(outputs[arm]["exact_lower"], outputs[arm]["majority"], matched_gold) for arm in ARMS},
    }


def write_items(path, rows, terminal, holdout):
    """Publish one row per question, with blank decisions when a fourth report is missing."""
    columns = ["item_id", "original_split", "gold", "three_report_decision"]
    for arm in ARMS:
        columns.extend(f"{arm}_{field}" for field in ("status", "fourth_count", "decision"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(rows):
            item = row["item_id"]
            prefix = [int(row[key]) for key in ("uniform", "random", "recency")]
            public = {"item_id": item, "original_split": "held_out" if i in holdout else "tuning",
                      "gold": row["gold"], "three_report_decision": predictions(prefix)["exact_lower"]}
            for arm in ARMS:
                status, count = terminal.get((item, arm), ("not_recorded", None))
                public[f"{arm}_status"] = status
                public[f"{arm}_fourth_count"] = count
                public[f"{arm}_decision"] = predictions([*prefix, count])["exact_lower"] if count is not None else None
            writer.writerow(public)


def historical_holdout(rows):
    """Summarize the original four reports independently of the new fourth calls."""
    correct = Counter({rule: 0 for rule in ("first_uniform", "three_uniform", *RULES)})
    for row in rows:
        values = [int(row[key]) for key in ("uniform", "random", "recency", "change")]
        scores = predictions(values)
        if scores["archived_float"] != int(row["archived_mamc"]):
            raise ValueError(f"Original floating-point estimate differs for {row['item_id']}")
        scores["first_uniform"] = values[0]
        scores["three_uniform"] = predictions(values[:3])["exact_lower"]
        for rule, estimate in scores.items():
            correct[rule] += estimate == int(row["gold"])
    return {"n": len(rows), "correct_by_rule": correct}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+", help="one or more terminal fourth-call record files")
    parser.add_argument("--scope", choices=("holdout", "all"), default="holdout")
    parser.add_argument("--archived-csv", type=Path, help="use an extracted copy of data/action_count.csv")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS, help="public item IDs and four answer options")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--items-out", type=Path, help="write an anonymous row-level table for the selected cohort")
    args = parser.parse_args()
    rows, terminal, hashes = inputs(args.jsonl, args.archived_csv, args.annotations)
    indices = np.sort(np.random.default_rng(0).permutation(206)[103:])
    held_out = [rows[i] for i in indices]
    result = {"input_sha256": hashes, "primary_103": summarize(held_out, terminal),
              "historical_fourth_holdout_103": historical_holdout(held_out)}
    if args.scope == "all":
        result["secondary_206_including_tuning_half"] = summarize(rows, terminal)
    if args.items_out:
        if args.scope == "all":
            write_items(args.items_out, rows, terminal, set(indices))
        else:
            write_items(args.items_out, held_out, terminal, set(range(len(held_out))))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
