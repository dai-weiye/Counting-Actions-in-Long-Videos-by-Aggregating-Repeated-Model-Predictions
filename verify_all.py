"""Check the file manifest and reproduce the numerical studies without model access."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, "-B", script, *args], cwd=ROOT, check=True)


def compare(stored, generated):
    if (ROOT / stored).read_bytes() != generated.read_bytes():
        raise ValueError(f"Replay differs from {stored}")


def main():
    manifest = (ROOT / "SHA256SUMS").read_text().splitlines()
    for entry in manifest:
        expected, relative = entry.split("  ", 1)
        if sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"File digest differs for {relative}")
    print(f"Verified {len(manifest)} file digests.", flush=True)

    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        result = folder / "replay.json"
        items = folder / "items.csv"
        counts = "data/action_count.csv"
        run("reproduce.py", "--out", result)
        run("fourth_call/analyze.py", "fourth_call/terminal_counts.jsonl",
            "--archived-csv", counts, "--annotations", "fourth_call/options.json",
            "--scope", "all", "--out", result, "--items-out", items)
        compare("fourth_call/results.json", result)
        compare("fourth_call/paired_items.csv", items)
        run("repeatability/analyze_repeats.py", "repeatability/records.csv", "--out", result)
        compare("repeatability/results.json", result)

        run("open_weight_check/score.py", "--labels", counts, "--out", result, "--items-out", items)
        compare("open_weight_check/results.json", result)
        compare("open_weight_check/items.csv", items)
        run("open_weight_test/score.py", "--out", result)
        compare("open_weight_test/results.json", result)
        run("open_weight_seed/score_groups.py", "--design", "open_weight_seed/design.json",
            "--counts", "open_weight_seed/predictions.jsonl", "--csv", counts, "--out", result)
        compare("open_weight_seed/results.json", result)
        run("range_sensitivity/candidate_ranges.py", "--package", ROOT, "--out", result)
        compare("range_sensitivity/results.json", result)

        for study in ("recovery", "main"):
            prefix = f"seed_replication/{study}"
            complete = (ROOT / f"{prefix}_results.json").is_file()
            script = "seed_replication/analyze.py" if complete else "seed_replication/coverage.py"
            args = ["--csv", counts] if complete else []
            run(script, "--design", f"{prefix}_design.json", "--counts",
                f"{prefix}_terminal_counts.jsonl", *args, "--out", result)
            compare(f"{prefix}_{'results' if complete else 'coverage'}.json", result)
        run("seed_replication/coverage.py", "--design", "seed_replication/main_design.json",
            "--counts", "seed_replication/main_terminal_counts.jsonl", "--out", result)
        compare("seed_replication/main_coverage.json", result)
        run("api_range_sensitivity/api_candidate_ranges.py", "--package", ROOT, "--out", result)
        compare("api_range_sensitivity/results.json", result)

        test_args = []
        if (ROOT / "external_eval/results.json").is_file():
            run("external_eval/verify_release.py")
            run("external_eval/score_test.py", "--out", result)
            compare("external_eval/results.json", result)
            test_args = ["--test-items", "external_eval/items.json",
                         "--test-calls", "external_eval/calls.jsonl"]
        else:
            output = subprocess.check_output(
                [sys.executable, "-B", "external_eval/verify_checkpoint.py"], cwd=ROOT)
            if json.loads(output) != json.loads((ROOT / "external_eval/coverage.json").read_bytes()):
                raise ValueError("Replay differs from the official test coverage ledger.")

        for domain, suffix in (("fixed", ""), ("offered", "_offered")):
            run("exact_stopping/replay.py", "--dev-csv", counts, "--dev-options",
                "fourth_call/options.json", "--open-weight-calls", "open_weight_check/predictions.jsonl",
                "--open-weight-test-cohort", "open_weight_test/cohort.json", "--open-weight-test-calls",
                "open_weight_test/predictions.jsonl", *test_args, "--candidate-domain", domain,
                "--out", result, "--items-out", items)
            compare(f"exact_stopping/results{suffix}.json", result)
            compare(f"exact_stopping/items{suffix}.csv", items)
        run("exact_stopping/verify.py", "--dev-options", "fourth_call/options.json",
            "--test-items", "open_weight_test/cohort.json")
        policies = folder / "policies"
        run("stopping_comparison/compare.py", "--package", ROOT, "--out-dir", policies)
        for name in ("results.json", "items.csv"):
            compare(f"stopping_comparison/{name}", policies / name)
        run("stopping_comparison/paired_stops.py", "--root", ROOT, "--out", result)
        compare("stopping_comparison/paired_stops.json", result)
        run("stopping_comparison/seed_groups.py", "--package", ROOT, "--out-dir", policies)
        for name in ("seed_groups.json", "seed_items.csv"):
            compare(f"stopping_comparison/{name}", policies / name)
        run("stopping_comparison/seed_horizons.py", "--package", ROOT, "--out", result)
        compare("stopping_comparison/seed_horizons.json", result)
        run("stopping_comparison/token_replay.py", "--package", ROOT, "--items",
            policies / "items.csv", "--usage", "stopping_usage/official_api_usage.jsonl",
            "--out", result)
        compare("stopping_comparison/token_results.json", result)
        run("closed_form/verify_closed_form.py", "--package", ROOT, "--out", result)
        compare("closed_form/results.json", result)
        run("stopping_comparison/reasoning_replay.py", "--package", ROOT, "--out", result)
        compare("stopping_comparison/reasoning_results.json", result)
        held_out = folder / "held_out"
        run("held_out_errors/analyze.py", "--csv", counts, "--out-dir", held_out)
        for name in ("results.json", "items.csv"):
            compare(f"held_out_errors/{name}", held_out / name)
        run("figures/verify_case.py", "--package", ROOT, "--out", result)
        compare("figures/mechanism_case.json", result)
        run("published_baselines/verify.py", "--package", ROOT)
    print("All offline checks passed.")


if __name__ == "__main__":
    main()
