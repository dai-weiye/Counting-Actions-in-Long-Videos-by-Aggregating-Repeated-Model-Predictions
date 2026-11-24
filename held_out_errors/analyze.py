"""Reproduce the archived decoder's error changes on the original held-out half."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from itertools import product
import json
from math import comb
from pathlib import Path

import numpy as np


REPORTS = ("uniform", "random", "recency", "change")
CANDIDATES = np.arange(1, 6)
GRID = tuple(product((0.5, 0.8, 1.0, 1.3), (0.8, 1.0, 1.5, 2.0), (0.0, 0.4, 0.8, 1.5)))


def score(values, parameters):
    """Match the original left-to-right binary64 accumulation and candidate order."""
    alpha, beta, gamma = parameters
    totals = np.zeros((len(values), len(CANDIDATES)), dtype=np.float64)
    for report in values.T:
        r = report[:, None]
        totals += (-alpha * np.maximum(CANDIDATES - r, 0)
                   - beta * np.maximum(r - CANDIDATES, 0) + gamma * (r == CANDIDATES))
    return CANDIDATES[totals.argmax(axis=1)]


def metrics(estimate, gold):
    error = estimate - gold
    return {"correct": int(np.sum(error == 0)), "accuracy_pct": float(100 * np.mean(error == 0)),
            "mae": float(np.mean(np.abs(error))), "under": int(np.sum(error < 0)), "over": int(np.sum(error > 0)),
            "under_pct": float(100 * np.mean(error < 0)), "over_pct": float(100 * np.mean(error > 0))}


def compare(single, full, gold):
    one, four = metrics(single, gold), metrics(full, gold)
    return {"n": len(gold), "one_call": one, "archived_mamc": four,
            "difference": {"net_correct": four["correct"] - one["correct"],
                           "accuracy_pp": 100 * (four["correct"] - one["correct"]) / len(gold),
                           "mae": four["mae"] - one["mae"],
                           "under_pp": 100 * (four["under"] - one["under"]) / len(gold),
                           "over_pp": 100 * (four["over"] - one["over"]) / len(gold)}}


def paired_inference(single, full, gold, resamples, seed):
    delta = (full == gold).astype(int) - (single == gold).astype(int)
    wins, losses = int(np.sum(delta == 1)), int(np.sum(delta == -1))
    discordant = wins + losses
    p = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for start in range(0, resamples, 5000):
        end = min(start + 5000, resamples)
        draws[start:end] = 100 * rng.choice(delta, size=(end - start, len(gold))).mean(axis=1)
    return {"corrected": wins, "introduced": losses, "both_correct": int(np.sum((single == gold) & (full == gold))),
            "both_incorrect": int(np.sum((single != gold) & (full != gold))), "exact_mcnemar_p": p,
            "accuracy_difference_pp": 100 * (wins - losses) / len(gold),
            "paired_bootstrap_percentile95_pp": np.percentile(draws, [2.5, 97.5]).tolist(),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed}


def direction(error):
    return "under" if error < 0 else "over" if error > 0 else "correct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="the package's data/action_count.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")
    with args.csv.open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: r["item_id"] + ".mp4")
    if len(rows) != 206 or len({r["item_id"] for r in rows}) != 206:
        raise ValueError("Expected one record per original development question")
    all_reports = np.array([[int(r[k]) for k in REPORTS] for r in rows])
    all_gold = np.array([int(r["gold"]) for r in rows])
    permutation = np.random.default_rng(0).permutation(206)
    train, held = np.sort(permutation[:103]), np.sort(permutation[103:])
    parameters = max(GRID, key=lambda v: int(np.sum(score(all_reports[train], v) == all_gold[train])))
    archive = np.array([int(r["archived_mamc"]) for r in rows])
    np.testing.assert_array_equal(score(all_reports, parameters), archive, err_msg="Reconstructed archival output differs")
    reports, gold, full = all_reports[held], all_gold[held], archive[held]
    single = reports[:, 0]
    durations = np.array([float(rows[i]["duration_s"]) for i in held])
    strata = {f"gold_{k}": gold == k for k in CANDIDATES}
    strata.update(under_600s=durations < 600, at_least_600s=durations >= 600)
    individual = []
    for j, i in enumerate(held):
        one, four, target = int(single[j]), int(full[j]), int(gold[j])
        one_dir, four_dir = direction(one - target), direction(four - target)
        individual.append({"item_id": rows[i]["item_id"], "gold": target, "duration_s": durations[j],
                           **{k: int(v) for k, v in zip(REPORTS, reports[j])}, "archived_mamc": four,
                           "one_call_direction": one_dir, "archived_direction": four_dir,
                           "one_call_abs_error": abs(one - target), "archived_abs_error": abs(four - target),
                           "any_report_correct": bool(target in reports[j])})
    transition = Counter((r["one_call_direction"], r["archived_direction"]) for r in individual)
    directions = ("under", "correct", "over")
    paired = paired_inference(single, full, gold, args.resamples, args.bootstrap_seed)
    full_error = np.abs(full - gold)
    report = {
        "source_sha256": sha256(args.csv.read_bytes()).hexdigest(),
        "selection": {"split_seed": 0, "fitting_n": len(train), "held_out_n": len(held),
                      "parameters": list(parameters), "archived_predictions_verified": len(archive)},
        "overall": compare(single, full, gold), "paired": paired,
        "strata": {name: compare(single[m], full[m], gold[m]) for name, m in strata.items()},
        "figure_counts": {"questions": len(held), "changed_correctness": paired["corrected"] + paired["introduced"],
                          "residual_errors": int(np.sum(full != gold)), "corrected": paired["corrected"],
                          "introduced": paired["introduced"], "full_under": int(np.sum(full < gold)),
                          "full_over": int(np.sum(full > gold))},
        "directions_one_call_to_archived": {a: {b: transition[a, b] for b in directions} for a in directions},
        "archived_absolute_error_counts": {str(k): int(np.sum(full_error == k)) for k in range(int(max(full_error)) + 1)},
        "four_reports_disagree": int(np.sum(np.ptp(reports, axis=1) != 0)),
        "residual_undercount_with_a_correct_report": int(np.sum((full < gold) & np.any(reports == gold[:, None], axis=1))),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.out_dir / "items.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=individual[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(individual)


if __name__ == "__main__":
    main()
