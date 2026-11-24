"""Replay the 103 held-out uniform requests once using the fourth-call collector."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import threading

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fourth-call', type=Path, required=True, help='Published fourth_call directory')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--video-dir', action='append', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=5)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('paired_collection', args.fourth_call / 'collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.cv2.setNumThreads(1)
    key = os.getenv('MAMC_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
    if not key:
        raise SystemExit('Supply the project API credential through the environment')
    with (args.fourth_call / 'paired_items.csv').open() as file:
        planned = [r['item_id'] for r in csv.DictReader(file) if r['original_split'] == 'held_out']
    assert len(planned) == len(set(planned)) == 103
    annotations = {Path(r['video']).stem: r for r in json.loads(args.annotations.read_text())}
    hashes = {r['item_id']: r['payload_sha256'] for r in read_jsonl(args.fourth_call / 'selected_frames.jsonl') if r['arm'] == 'uniform'}
    run = args.run_dir
    (run / 'inflight').mkdir(parents=True, exist_ok=True)
    lock = (run / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read_jsonl(run / 'fourth_calls.jsonl')
    done = {(r['item_id'], r['arm']) for r in old}
    assert len(old) == len(done) and all(i in planned and a=='uniform' for i,a in done)
    stop = threading.Event()
    def replay(item):
        row = annotations[item]
        if (item, 'uniform') in done:
            return
        path = next((d / row['video'] for d in args.video_dir if (d / row['video']).is_file()), None)
        if path is None:
            print(f'{item}: source pending', flush=True)
            return
        module.collect_arm(row, path, 'uniform', key, run, done, stop, expected_payload_sha256=hashes[item], timeout=(180, 360))
    with ThreadPoolExecutor(args.jobs) as pool:
        work = [pool.submit(replay, item) for item in sorted(planned)]
        try:
            for task in as_completed(work):
                task.result()
        except BaseException:
            stop.set()
            for task in work:
                task.cancel()
            raise
if __name__ == '__main__':
    main()
