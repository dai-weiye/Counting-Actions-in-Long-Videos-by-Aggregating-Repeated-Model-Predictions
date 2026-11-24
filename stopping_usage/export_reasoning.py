"""Export reasoning token counts from the original official-test API receipts."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

SLOTS = ("U1", "U2", "U3", "C4")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def export(package, raw, out):
    release = package / "external_eval"
    usage_path = package / "stopping_usage/official_api_usage.jsonl"
    prior = json.loads((usage_path.parent / "provenance.json").read_text())
    input_path = release / "planned_slots.jsonl"
    if (
        digest(usage_path) != prior["usage_jsonl_sha256"]
        or digest(input_path) != prior["release_files_sha256"][input_path.name]
    ):
        raise ValueError(
            "The usage ledger or the frozen input ledger differs from its published digest"
        )
    cohort_path = release / "items.json"
    if digest(cohort_path) != prior["release_files_sha256"][cohort_path.name]:
        raise ValueError("The official-test roster differs from its published digest")
    cohort = json.loads(cohort_path.read_text())
    public_ids = {Path(row["video"]).stem: row["id"] for row in cohort}
    expected = {(item, slot) for item in public_ids.values() for slot in SLOTS}
    usage = json_lines(usage_path)
    planned = json_lines(input_path)
    existing = {(row["id"], row["slot"]): row for row in usage}
    inputs = {(row["id"], row["slot"]): row["payload_sha256"] for row in planned}
    if (
        len(expected) != 240
        or len(usage) != 240
        or len(planned) != 240
        or set(inputs) != expected
        or set(existing) != expected
    ):
        raise ValueError(
            "The official cohort, inputs and usage must agree on all 240 distinct slots"
        )

    ledger = {}
    source_hashes = {}
    for slot in SLOTS:
        receipt_path = raw / slot / "fourth_calls.jsonl"
        source_hashes[slot] = digest(receipt_path)
        if source_hashes[slot] != prior["source_receipt_files_sha256_by_slot"][slot]:
            raise ValueError(f"The {slot} source differs from the original usage export")
        for receipt in json_lines(receipt_path):
            item = public_ids[receipt["item_id"]]
            key = (item, slot)
            hashed = inputs[key]
            if (
                key in ledger
                or receipt["http_status"] != 200
                or receipt["payload_sha256"] != hashed
                or existing[key]["payload_sha256"] != hashed
            ):
                raise ValueError(f"Duplicate or mismatched receipt for {item}/{slot}")
            observed = receipt["usage"]
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if observed[field] != existing[key][field]:
                    raise ValueError(f"{item}/{slot} disagrees with the original {field}")
            count = observed["completion_tokens_details"]["reasoning_tokens"]
            if type(count) is not int or not 0 <= count <= observed["completion_tokens"]:
                raise ValueError(f"Invalid reasoning count for {item}/{slot}")
            ledger[key] = {
                "id": item,
                "slot": slot,
                "payload_sha256": hashed,
                "reasoning_tokens": count,
            }
    if set(ledger) != expected:
        raise ValueError("Missing reasoning counters")

    out.mkdir(parents=True, exist_ok=True)
    target = out / "reasoning_usage.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for key in existing:
            stream.write(json.dumps(ledger[key], separators=(",", ":")) + "\n")
    provenance = {
        "response_field": "usage.completion_tokens_details.reasoning_tokens",
        "questions": len(cohort),
        "slots": len(ledger),
        "total_reasoning_tokens": sum(row["reasoning_tokens"] for row in ledger.values()),
        "ledger_sha256": digest(target),
        "original_usage_sha256": digest(usage_path),
        "planned_slots_sha256": digest(input_path),
        "source_receipt_sha256_by_slot": source_hashes,
    }
    (out / "reasoning_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", required=True, type=Path, help="Extracted reproducibility package"
    )
    parser.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Private original receipts, in one directory per slot",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory for the anonymous ledger and its provenance",
    )
    args = parser.parse_args()
    export(args.package, args.raw, args.out)
