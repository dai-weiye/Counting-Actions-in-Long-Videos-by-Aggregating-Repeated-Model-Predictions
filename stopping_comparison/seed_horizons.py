"""Replay four- and eight-call budgets on the same frozen 32-video random-frame panel."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path


LEDGERS = {
    "deepseek": "seed_replication/main_terminal_counts.jsonl",
    "qwen": "open_weight_seed/predictions.jsonl",
}
GOLD = "data/action_count.csv"
CHOICES = "fourth_call/options.json"
PANEL = "open_weight_check/cohort.json"
ARCHIVE = "stopping_comparison/seed_groups.json"
POLICIES = ("full", "exact", "two_agree", "three_agree")
GROUPS = {4: [(g,) for g in range(1, 7)], 8: [(1, 2), (3, 4), (5, 6)]}


def score(report, candidate):
    return 0 if report is None else -2 * abs(report - candidate) + int(report == candidate)


@lru_cache(maxsize=None)
def worst_leads(choices, candidates):
    return {
        (leader, rival): min(score(x, leader) - score(x, rival) for x in (*choices, None))
        for leader in candidates for rival in candidates if leader != rival
    }


def certify(prefix, remaining, choices, candidates):
    if all(r is None for r in prefix):
        return remaining == 0, None
    totals = {c: sum(score(r, c) for r in prefix) for c in candidates}
    leader = max(candidates, key=totals.get)
    worst = worst_leads(choices, candidates)
    for rival in candidates:
        if leader != rival:
            margin = totals[leader] - totals[rival] + remaining * worst[leader, rival]
            if margin < 0 or (margin == 0 and rival < leader):
                return False, None
    return True, leader


def aggregate(reports, candidates):
    observed = [r for r in reports if r is not None]
    if not observed:
        return None
    # Independent batch form. Minimize twice absolute deviation minus exact matches.
    return min(candidates, key=lambda c: (2 * sum(abs(c - r) for r in observed) - observed.count(c), c))


def earliest_safe(reports, choices, candidates):
    """Verify the returned certificate and a counterexample at every earlier prefix."""
    for t in range(1, len(reports) + 1):
        prefix = reports[:t]
        left = len(reports) - t
        certified, prediction = certify(prefix, left, choices, candidates)
        current = aggregate(prefix, candidates)
        # A blank-only suffix keeps the current decision. Addition is order-invariant,
        # so enumerating multisets covers all legal suffixes, including blanks.
        suffixes = combinations_with_replacement((*choices, None), left)
        invariant = all(aggregate(prefix + s, candidates) == current for s in suffixes)
        if certified != invariant or (certified and prediction != current):
            raise ValueError(f"Certificate mismatch at {prefix} with {left} slots left")
        if certified:
            return t, prediction
    raise AssertionError("No certified prefix at the full horizon")


def replay(reports, choices, candidates):
    full = aggregate(reports, candidates)
    outputs = {"full": (len(reports), full), "exact": earliest_safe(reports, choices, candidates)}
    if outputs["exact"][1] != full:
        raise ValueError("The certificate disagrees with the observed completion")
    for size, name in [(2, "two_agree"), (3, "three_agree")]:
        prefix = reports[:size]
        use = size if prefix[0] is not None and len(set(prefix)) == 1 else len(reports)
        outputs[name] = use, aggregate(reports[:use], candidates)
    return outputs


def run(root):
    manifest = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (root / "SHA256SUMS").read_text().splitlines())}
    sources = sorted({*LEDGERS.values(), GOLD, CHOICES, PANEL, ARCHIVE})
    for name in sources:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f"File differs from the archive manifest: {name}")
    with (root / GOLD).open() as f:
        gold = {r["item_id"]: int(r["gold"]) for r in csv.DictReader(f)}
    options = {Path(r["video"]).stem: tuple(sorted(set(map(int, r["candidates"]))))
               for r in json.loads((root / CHOICES).read_text())}
    panel = sorted(Path(r["video"]).stem for r in json.loads((root / PANEL).read_text()))
    if len(panel) != 32 or len(set(panel)) != 32:
        raise ValueError("Expected 32 distinct videos")
    archived = json.loads((root / ARCHIVE).read_text())["by_model"]
    result = {"videos": len(panel), "group_order": GROUPS, "by_model": {},
              "source_sha256": {name: manifest[name] for name in sources}}
    for model, file in LEDGERS.items():
        raw = [json.loads(line) for line in (root / file).read_text().splitlines()]
        slots = {(r["item_id"], r["group"], r["call"]): r["count"] for r in raw}
        if len(raw) != len(slots) or set(slots) != set(product(panel, range(1, 7), range(1, 5))):
            raise ValueError(f"Invalid slot coverage for {model}")
        for r in raw:
            v = r["count"]
            if (r["status"] == "ok") != (v is not None) or (
                v is not None and (type(v) is not int or v not in options[r["item_id"]])
            ):
                raise ValueError(f"Invalid terminal response for {model}")
        result["by_model"][model] = {}
        for domain in ("fixed", "offered"):
            result["by_model"][model][domain] = {}
            for horizon, blocks in GROUPS.items():
                overall = {name: Counter() for name in POLICIES}
                changed = {name: [] for name in ("two_agree", "three_agree")}
                for groups in blocks:
                    by_group = {name: Counter() for name in POLICIES}
                    for item in panel:
                        reports = tuple(slots[item, g, j] for g in groups for j in range(1, 5))
                        choices = options[item]
                        candidates = tuple(range(1, 6)) if domain == "fixed" else choices
                        if gold[item] not in choices:
                            raise ValueError(f"Unlisted gold count for {item}")
                        outputs = replay(reports, choices, candidates)
                        full = outputs["full"][1]
                        for name, (calls, prediction) in outputs.items():
                            by_group[name].update(n=1, calls=calls, matches_full=int(prediction == full),
                                                  correct=int(prediction == gold[item]), answered=int(prediction is not None))
                            if name in changed and prediction != full and horizon == 8:
                                changed[name].append({"id": item, "groups": groups, "reports": reports,
                                                      "early": prediction, "full": full, "gold": gold[item]})
                    for name in POLICIES:
                        overall[name].update(by_group[name])
                        if horizon == 4:
                            source_name = "fixed_4" if name == "full" else name
                            ref = archived[model][domain]["by_group"][str(groups[0])][source_name]
                            for key, value in by_group[name].items():
                                if value != ref["matches_four" if key == "matches_full" else key]:
                                    raise ValueError(f"Archived K=4 result differs for {model}, {domain}, {groups}, {name}, {key}")
                summary = {"policies": overall}
                if horizon == 8:
                    summary["changed_by_policy"] = changed
                result["by_model"][model][domain][str(horizon)] = summary
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.package), indent=2) + "\n")
