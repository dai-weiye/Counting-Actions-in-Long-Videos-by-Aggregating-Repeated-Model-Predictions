"""Check the transcribed Action Count scores against the independent source index."""

import argparse
import csv
import json
from pathlib import Path


def verify(package):
    directory = package / "published_baselines"
    sources = json.loads((directory / "sources.json").read_text())
    with (directory / "reported_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    reported = {row["id"]: row for row in rows}
    indexed = {row["id"]: row for row in sources["baselines"]}
    if len(rows) != 6 or len(reported) != 6 or reported.keys() != indexed.keys():
        raise ValueError("Expected the six independently verified published systems")
    verified_scores = 0
    for key, row in reported.items():
        source = indexed[key]
        if row["method"] != source["method"]:
            raise ValueError(f"Method mismatch for {key}")
        for split in ("dev", "test"):
            value = row[f"{split}_action_count_pct"]
            reference = source["results"].get(split)
            if reference is None:
                if value:
                    raise ValueError(f"Unreported {split} score for {key}")
                continue
            if not value or float(value) != reference["accuracy_pct"]:
                raise ValueError(f"Score mismatch for {key}/{split}")
            table = sources["result_tables"][reference["table_id"]]
            if not table["url"].startswith("https://") or table["pdf_page"] < 1:
                raise ValueError(f"Missing original table for {key}/{split}")
            verified_scores += 1
    if verified_scores != 9:
        raise ValueError("Expected six development and three test scores")
    print("Verified six published systems, six development scores and three test scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    verify(args.package)
