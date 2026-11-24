"""Replay observed reasoning-token usage for each prefix policy on the official test."""

import argparse
import csv
import json
from hashlib import sha256
from itertools import product
from pathlib import Path

SLOTS = ("U1", "U2", "U3", "C4")
DOMAINS = ("fixed", "offered")
POLICIES = ("fixed_1", "fixed_2", "fixed_3", "fixed_4", "three_agree", "two_agree", "exact")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_slots(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lookup = {(r["id"], r["slot"]): r for r in rows}
    ids = {r["id"] for r in rows}
    if len(rows) != 240 or len(ids) != 60 or set(lookup) != set(product(ids, SLOTS)):
        raise ValueError(f"{path.name} must contain each of 240 official-test slots exactly once")
    return lookup


def replay(package, reasoning_path):
    usage_path = package / "stopping_usage/official_api_usage.jsonl"
    planned_path = package / "external_eval/planned_slots.jsonl"
    item_path = package / "stopping_comparison/items.csv"
    prior_path = package / "stopping_comparison/token_results.json"
    prior = json.loads(prior_path.read_text())
    original_inputs = prior["input_sha256"]
    for path, key in (
        (usage_path, "usage_ledger"),
        (planned_path, "planned_slots"),
        (item_path, "policy_items"),
    ):
        if digest(path) != original_inputs[key]:
            raise ValueError(f"{path.name} differs from the published token replay")
    provenance = json.loads((reasoning_path.parent / "reasoning_provenance.json").read_text())
    if (
        digest(reasoning_path) != provenance["ledger_sha256"]
        or digest(usage_path) != provenance["original_usage_sha256"]
        or digest(planned_path) != provenance["planned_slots_sha256"]
    ):
        raise ValueError(
            "The reasoning ledger and provenance do not match the original usage or planned inputs"
        )
    planned, usage, reasoning = map(read_slots, (planned_path, usage_path, reasoning_path))
    if set(usage) != set(planned) or set(reasoning) != set(planned):
        raise ValueError("The three ledgers disagree on official-test slots")
    for key, row in reasoning.items():
        hashed = planned[key]["payload_sha256"]
        count = row["reasoning_tokens"]
        if row["payload_sha256"] != hashed or usage[key]["payload_sha256"] != hashed:
            raise ValueError(f"Unmatched frozen request for {key}")
        if type(count) is not int or not 0 <= count <= usage[key]["completion_tokens"]:
            raise ValueError(f"Invalid reasoning-token count for {key}")
    full = sum(row["reasoning_tokens"] for row in reasoning.values())
    if full != provenance["total_reasoning_tokens"] or full <= 0:
        raise ValueError("The full token total disagrees with provenance")

    ids = sorted({key[0] for key in planned})
    lengths = {}
    with item_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "test":
                continue
            policy_key = (row["domain"], row["policy"], row["item_id"])
            n = int(row["calls"])
            if policy_key in lengths or n not in range(1, 5):
                raise ValueError(f"Duplicate or invalid test prefix for {policy_key}")
            lengths[policy_key] = n
    if set(lengths) != set(product(DOMAINS, POLICIES, ids)):
        raise ValueError("Missing official-test policy prefixes")

    totals = {}
    for domain in DOMAINS:
        totals[domain] = {}
        for policy in POLICIES:
            kept = [(item, slot) for item in ids for slot in SLOTS[: lengths[domain, policy, item]]]
            old = prior["replayed_usage_by_domain"][domain][policy]
            if (
                len(kept) != old["calls"]
                or sum(usage[k]["completion_tokens"] for k in kept) != old["completion_tokens"]
            ):
                raise ValueError(
                    f"The retained prefix differs from the original token replay for {domain}/{policy}"
                )
            total = sum(reasoning[k]["reasoning_tokens"] for k in kept)
            totals[domain][policy] = {
                "calls": len(kept),
                "reasoning_tokens": total,
                "saved_vs_four": full - total,
                "saved_pct_vs_four": 100 * (full - total) / full,
            }
        if totals[domain]["fixed_4"]["reasoning_tokens"] != full:
            raise ValueError(f"The four-slot policy omits a reasoning counter in {domain}")
    return {
        "questions": len(ids),
        "observed_api_slots": len(reasoning),
        "full_reasoning_tokens": full,
        "input_sha256": {
            **original_inputs,
            "reasoning_usage": digest(reasoning_path),
            "original_token_results": digest(prior_path),
        },
        "by_domain": totals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", required=True, type=Path, help="Extracted reproducibility package"
    )
    parser.add_argument(
        "--reasoning",
        type=Path,
        help="Override the location of stopping_usage/reasoning_usage.jsonl",
    )
    parser.add_argument("--out", required=True, type=Path, help="JSON output path")
    args = parser.parse_args()
    source = args.reasoning or args.package / "stopping_usage/reasoning_usage.jsonl"
    results = replay(args.package, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
