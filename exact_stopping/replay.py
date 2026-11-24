"""Replay exact early stopping with a fixed four-report counting rule."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from functools import cache
from numbers import Integral
from pathlib import Path

SLOTS = ("U1", "U2", "U3", "C4")
CANDIDATES = range(1, 6)


def score(report, candidate):
    if report is None:
        return 0
    return -2 * abs(report - candidate) + int(report == candidate)


def ordered_counts(values):
    values = tuple(values)
    if not values or any(
        not isinstance(v, Integral) or isinstance(v, bool) for v in values
    ):
        raise ValueError(
            "A candidate or option set must contain at least one integer count"
        )
    return tuple(sorted(set(map(int, values))))


def aggregate(reports, candidates=CANDIDATES):
    candidates = ordered_counts(candidates)
    if all(report is None for report in reports):
        return None
    # max returns the first maximizing candidate, yielding the smaller count on ties.
    return max(candidates, key=lambda c: sum(score(r, c) for r in reports))


@cache
def worst_single_report_margins(choices, candidates):
    outcomes = (*choices, None)
    return {
        (leader, rival): min(score(r, leader) - score(r, rival) for r in outcomes)
        for leader in candidates
        for rival in candidates
        if rival != leader
    }


def stop_when_fixed(prefix, choices, candidates=CANDIDATES):
    """Return (decision is fixed, decision) after observing this prefix."""
    if len(prefix) > 4:
        raise ValueError("Expected at most four reports")
    choices = ordered_counts(choices)
    candidates = ordered_counts(candidates)
    if any(v is not None and v not in choices for v in prefix):
        raise ValueError("A report is absent from the listed options")
    remaining = 4 - len(prefix)
    if all(report is None for report in prefix):
        return remaining == 0, None

    totals = {c: sum(score(r, c) for r in prefix) for c in candidates}
    leader = max(candidates, key=totals.get)
    margins = worst_single_report_margins(choices, candidates)
    for rival in candidates:
        if rival == leader:
            continue
        final_margin = (
            totals[leader] - totals[rival] + remaining * margins[leader, rival]
        )
        if final_margin < 0 or (final_margin == 0 and rival < leader):
            return False, None
    return True, leader


def count_sequential(choices, source, candidates=CANDIDATES):
    """Return (count, number of requests), acquiring each ordered slot only when needed.

    A callable receives a slot name (U1, U2, U3 or C4). An iterable may
    instead generate the reports lazily. Supply None for an unusable final.
    """
    choices = ordered_counts(choices)
    candidates = ordered_counts(candidates)
    iterator = None if callable(source) else iter(source)
    prefix = []
    for slot in SLOTS:
        if iterator is None:
            report = source(slot)
        else:
            try:
                report = next(iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"The report stream ended before required slot {slot}"
                ) from exc
        prefix.append(report)
        fixed, decision = stop_when_fixed(prefix, choices, candidates)
        if fixed:
            return decision, len(prefix)
    raise AssertionError("Four observed slots must determine a unique decision")


def replay(item, options, reports, candidates=CANDIDATES):
    options = ordered_counts(options)
    candidates = ordered_counts(candidates)
    if len(reports) != 4:
        raise ValueError("Expected exactly four report slots")
    if any(x is not None and x not in options for x in reports):
        raise ValueError(f"Report is absent from the offered options for {item}")
    full = aggregate(reports, candidates)
    early, used = count_sequential(options, iter(reports), candidates)
    if early != full:
        raise AssertionError(f"Certified decision differs for {item}")
    return {
        "item_id": item,
        "four_usable": all(x is not None for x in reports),
        "used_calls": used,
        "saved_calls": 4 - used,
        "early_count": early,
        "four_slot_count": full,
    }


def summarize(rows):
    used = sum(r["used_calls"] for r in rows)
    n = len(rows)
    stops = Counter(r["used_calls"] for r in rows)
    return {
        "n": n,
        "planned_calls": 4 * n,
        "calls_to_fixed_decision": used,
        "calls_saved": 4 * n - used,
        "saved_pct": 100 * (4 * n - used) / (4 * n) if n else None,
        "equal_decisions": sum(r["early_count"] == r["four_slot_count"] for r in rows),
        "stopped_after_calls": {str(k): stops[k] for k in range(1, 5)},
        "four_usable": sum(r["four_usable"] for r in rows),
        "unanswered_after_four": sum(r["four_slot_count"] is None for r in rows),
    }


def run(opts):
    import numpy as np

    def score_item(item, options, reports):
        domain = options if opts.candidate_domain == "offered" else CANDIDATES
        return replay(item, options, reports, domain)

    with opts.dev_csv.open() as f:
        dev = sorted(csv.DictReader(f), key=lambda r: r["item_id"] + ".mp4")
    choices = {
        Path(r["video"]).stem: tuple(map(int, r["candidates"]))
        for r in json.loads(opts.dev_options.read_text())
    }
    ids = [r["item_id"] for r in dev]
    if len(dev) != 206 or set(ids) != set(choices):
        raise ValueError("Invalid development-set inputs")
    holdout = set(np.random.default_rng(0).permutation(206)[103:].tolist())
    results = {}
    by_split = []
    dev_rows = []
    for i, r in enumerate(dev):
        item = r["item_id"]
        v = [int(r[k]) for k in ("uniform", "random", "recency", "change")]
        out = score_item(item, choices[item], v)
        dev_rows.append(out)
        by_split.append({"split": "dev_held_out" if i in holdout else "dev_fit", **out})
    results["development_all"] = summarize(dev_rows)
    results["development_held_out"] = summarize([dev_rows[i] for i in sorted(holdout)])
    files = {opts.dev_csv.name: opts.dev_csv, opts.dev_options.name: opts.dev_options}
    test_inputs = [
        (opts.test_items, opts.test_calls, "id", "official_test", "test"),
        (
            opts.open_weight_test_cohort,
            opts.open_weight_test_calls,
            "item_id",
            "open_weight_official_test",
            "open_weight_test",
        ),
    ]
    for item_path, record_path, id_key, study, split in test_inputs:
        if bool(item_path) != bool(record_path):
            raise ValueError(f"Both {study} inputs are required")
        if not item_path:
            continue
        items = json.loads(item_path.read_text())
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        lookup = {(r[id_key], r["slot"]): r for r in records}
        if (
            len(items) != 60
            or len(lookup) != len(records)
            or set(lookup) != {(q["id"], slot) for q in items for slot in SLOTS}
        ):
            raise ValueError(
                f"Expected 60 {study} questions with four terminal slots each"
            )
        rows = []
        for q in sorted(items, key=lambda r: r["id"]):
            item = q["id"]
            slots = [lookup[item, slot] for slot in SLOTS]
            if any((r["status"] == "ok") != (r["count"] is not None) for r in slots):
                raise ValueError("Inconsistent terminal response")
            row = score_item(
                item, tuple(map(int, q["options"])), [r["count"] for r in slots]
            )
            rows.append(row)
            by_split.append({"split": split, **row})
        results[study + "_all"] = summarize(rows)
        results[study + "_four_usable"] = summarize(
            [r for r in rows if r["four_usable"]]
        )
        files[study + "_cohort"] = item_path
        files[study + "_calls"] = record_path
    if opts.open_weight_calls:
        records = [
            json.loads(line) for line in opts.open_weight_calls.read_text().splitlines()
        ]
        lookup = {(row["item_id"], row["slot"]): row for row in records}
        item_ids = sorted({row["item_id"] for row in records})
        if (
            len(item_ids) != 32
            or len(lookup) != len(records)
            or set(lookup) != {(item, slot) for item in item_ids for slot in SLOTS}
        ):
            raise ValueError(
                "Expected one call per slot on all 32 open-weight study items"
            )
        rows = []
        for item in item_ids:
            values = [lookup[item, slot]["count"] for slot in SLOTS]
            row = score_item(item, choices[item], values)
            rows.append(row)
            by_split.append({"split": "open_weight_held_out", **row})
        results["open_weight_held_out"] = summarize(rows)
        files["open_weight_held_out_calls"] = opts.open_weight_calls
    results["input_sha256"] = {
        name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()
    }
    opts.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    with opts.items_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=by_split[0])
        writer.writeheader()
        writer.writerows(by_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-domain",
        choices=("fixed", "offered"),
        default="fixed",
        help="Use 1 through 5 (fixed) or the current question's listed counts (offered)",
    )
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--dev-options", type=Path, required=True)
    parser.add_argument("--test-items", type=Path)
    parser.add_argument("--test-calls", type=Path)
    parser.add_argument("--open-weight-test-cohort", type=Path)
    parser.add_argument("--open-weight-test-calls", type=Path)
    parser.add_argument("--open-weight-calls", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items-out", type=Path, required=True)
    run(parser.parse_args())
