"""Sum observed official API token usage for the slots retained by each stopping policy."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from compare import POLICIES, file_digest


SLOTS = ("U1", "U2", "U3", "C4")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read_slots(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    lookup = {(r["id"], r["slot"]): r for r in rows}
    ids = {r["id"] for r in rows}
    if len(ids) != 60 or len(rows) != 240 or set(lookup) != {(i, s) for i in ids for s in SLOTS}:
        raise ValueError(f"Expected four distinct slots for each of the 60 questions in {path}")
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True, help="anonymous per-slot API usage ledger")
    parser.add_argument("--items", type=Path, required=True, help="items.csv produced by compare.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    planned_path = args.package / "external_eval/planned_slots.jsonl"
    planned, usage = read_slots(planned_path), read_slots(args.usage)
    if {k: v["payload_sha256"] for k, v in planned.items()} != {k: v["payload_sha256"] for k, v in usage.items()}:
        raise ValueError("The token receipts do not match the frozen payloads")
    for key, row in usage.items():
        if any(type(row[f]) is not int or row[f] < 0 for f in TOKEN_FIELDS):
            raise ValueError(f"Invalid token count in {key}")
        if row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
            raise ValueError(f"Token counters do not balance in {key}")
    ids = {k[0] for k in planned}
    with args.items.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "test"]
    degrees = {(r["domain"], r["policy"], r["item_id"]): int(r["calls"]) for r in rows}
    keys = {(d, p, i) for d in ("fixed", "offered") for p in POLICIES for i in ids}
    if len(rows) != len(keys) or degrees.keys() != keys or any(k not in range(1, 5) for k in degrees.values()):
        raise ValueError("Missing or duplicate test policy prefix")
    full = {field: sum(row[field] for row in usage.values()) for field in TOKEN_FIELDS}
    results = {}
    for domain in ("fixed", "offered"):
        results[domain] = {}
        for policy in POLICIES:
            total = Counter()
            for item in sorted(ids):
                used = degrees[domain, policy, item]
                total["calls"] += used
                for slot in SLOTS[:used]:
                    for field in TOKEN_FIELDS:
                        total[field] += usage[item, slot][field]
            results[domain][policy] = {**total, "tokens_saved_vs_four": full["total_tokens"] - total["total_tokens"],
                                       "saved_pct_vs_four": 100 * (full["total_tokens"] - total["total_tokens"]) / full["total_tokens"]}
        if any(results[domain]["fixed_4"][f] != full[f] for f in TOKEN_FIELDS):
            raise ValueError(f"Four-slot receipts do not reconcile for {domain}")
    out = {"questions": len(ids), "observed_api_slots": len(usage), "full_four_slot_usage": full,
           "input_sha256": {"planned_slots": file_digest(planned_path), "usage_ledger": file_digest(args.usage),
                            "policy_items": file_digest(args.items)}, "replayed_usage_by_domain": results}
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
