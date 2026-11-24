"""Run the fixed four-slot MLVU Test experiment on a label-free manifest."""

import argparse
import fcntl
import importlib.util
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")


def run(opts):
    key = os.environ.get("MAMC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set MAMC_API_KEY for the official model endpoint.")
    spec = importlib.util.spec_from_file_location(
        "fourth_collector", opts.protocol_root / "fourth_call/collect.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # The sole live transport override extends the timeout for multi-megabyte frame uploads.
    import requests

    class Transport:
        RequestException = requests.RequestException

        @staticmethod
        def post(url, **kwargs):
            kwargs["timeout"] = (180, 360)
            return requests.post(url, **kwargs)

    driver.requests = Transport
    driver.cv2.setNumThreads(1)
    opts.run_dir.mkdir(exist_ok=True, parents=True)
    lock = (opts.run_dir / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    while not opts.questions.is_file():
        time.sleep(2)
    manifest = json.loads(opts.questions.read_text())
    if len(manifest) != 60 or any("answer" in r or "gold" in r for r in manifest):
        raise ValueError("Expected 60 label-free question records")
    done = {}
    for slot in SLOTS:
        folder = opts.run_dir / slot
        (folder / "inflight").mkdir(parents=True, exist_ok=True)
        ledger = folder / "fourth_calls.jsonl"
        records = (
            [json.loads(i) for i in ledger.read_text().splitlines()]
            if ledger.exists()
            else []
        )
        done[slot] = {(r["item_id"], r["arm"]) for r in records}
        if len(done[slot]) != len(records):
            raise ValueError(f"Duplicate slot {slot}")
        for item, arm in done[slot]:
            (folder / "inflight" / f"{item}-{arm}.json").unlink(missing_ok=True)

    def work(index, row):
        slot_order = SLOTS[index % 4 :] + SLOTS[: index % 4]
        video = opts.videos / row["video"]
        for slot in slot_order:
            if stop.is_set():
                return
            arm = "change" if slot == "C4" else "uniform"
            driver.collect_arm(
                row, video, arm, key, opts.run_dir / slot, done[slot], stop
            )

    def is_done(row):
        return all(
            (Path(row["video"]).stem, "change" if s == "C4" else "uniform") in done[s]
            for s in SLOTS
        )

    todo = {i: r for i, r in enumerate(manifest) if not is_done(r)}
    active = set()
    with ThreadPoolExecutor(opts.jobs) as pool:
        try:
            while todo or active:
                for index, row in list(todo.items()):
                    if len(active) == opts.jobs:
                        break
                    if (opts.videos / row["video"]).is_file():
                        active.add(pool.submit(work, index, row))
                        del todo[index]
                if active:
                    finished, active = wait(
                        active, timeout=3, return_when=FIRST_COMPLETED
                    )
                    for future in finished:
                        future.result()
                elif todo:
                    time.sleep(3)
        except BaseException:
            stop.set()
            for future in active:
                future.cancel()
            raise
    print("All assigned questions visited.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.json")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--run-dir", type=Path, required=True)
    package = (
        (ROOT.parents[1] / "paper/ICME2027_REPRODUCIBILITY")
        if not (ROOT.parent / "fourth_call").is_dir()
        else ROOT.parent
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=package,
        help="Path to the extracted reproducibility package",
    )
    run(parser.parse_args())
