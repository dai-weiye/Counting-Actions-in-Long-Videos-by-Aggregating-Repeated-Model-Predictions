"""Reconstruct both visual inputs for every test item without opening answer labels."""

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")


def hash_file(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def write_lines(path, rows):
    temp = path.with_suffix(".pending")
    temp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=HERE / "questions.json")
    p.add_argument("--out", type=Path, required=True)
    package = (
        HERE.parent
        if (HERE.parent / "fourth_call").is_dir()
        else HERE.parents[1] / "paper/ICME2027_REPRODUCIBILITY"
    )
    p.add_argument("--protocol-root", type=Path, default=package)
    p.add_argument(
        "--receipts",
        type=Path,
        help="Optional raw run directory for validating existing payloads",
    )
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    questions = json.loads(args.questions.read_text())
    assert len(questions) == 60 and all(
        set(q).isdisjoint({"answer", "gold", "label"}) for q in questions
    )
    args.out.mkdir(exist_ok=True, parents=True)
    part = args.out / "input_cache"
    part.mkdir(exist_ok=True)
    sampler_path = args.protocol_root / "fourth_call/collect.py"
    spec = importlib.util.spec_from_file_location("official_sampler", sampler_path)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    sampler.cv2.setNumThreads(1)
    dep = {
        "questions": hash_file(args.questions),
        "sampler": hash_file(sampler_path),
        "original_protocol": hash_file(args.protocol_root / "inference_protocol.py"),
        "test_collector": hash_file(HERE / "collect_test.py"),
        "analysis_plan": hash_file(HERE / "protocol.md"),
    }
    identity = sha256(json.dumps(dep, sort_keys=True).encode()).hexdigest()

    def construct(q):
        cached = part / (q["id"] + ".json")
        if cached.exists():
            data = json.loads(cached.read_text())
            if data["lock"] == identity:
                return data["rows"]
        video = args.videos / q["video"]
        video_sha = hash_file(video)
        prompt = sampler.original.make_prompt(q["question"], q["candidates"])
        outputs = []
        for arm in ("uniform", "change"):
            images, indices = sampler.sample(video, arm)
            payload = {
                "model": "deepseek-flash",
                "temperature": 0.0,
                "max_tokens": 8192,
                "thinking": {"type": "enabled"},
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *images],
                    }
                ],
            }
            data = json.dumps(payload, separators=(",", ":")).encode()
            outputs.append(
                {
                    "id": q["id"],
                    "video": video.name,
                    "video_sha256": video_sha,
                    "arm": arm,
                    "frame_indices": indices,
                    "payload_sha256": sha256(data).hexdigest(),
                    "payload_bytes": len(data),
                }
            )
            del images, payload, data
        temp = cached.with_suffix(".tmp")
        temp.write_text(json.dumps({"lock": identity, "rows": outputs}))
        temp.replace(cached)
        return outputs

    with ThreadPoolExecutor(args.jobs) as pool:
        frames = [r for rows in pool.map(construct, questions) for r in rows]
    indexed = {(r["id"], r["arm"]): r for r in frames}
    assert len(indexed) == 120
    planned = [
        {
            "id": q["id"],
            "slot": slot,
            "payload_sha256": indexed[q["id"], "change" if slot == "C4" else "uniform"][
                "payload_sha256"
            ],
        }
        for q in questions
        for slot in SLOTS
    ]
    assert len(planned) == 240
    if args.receipts:
        key_by_name = {Path(q["video"]).stem: q["id"] for q in questions}
        for f in args.receipts.glob("*/fourth_calls.jsonl"):
            for line in f.read_text().splitlines():
                obs = json.loads(line)
                want = indexed[key_by_name[obs["item_id"]], obs["arm"]]
                assert (
                    obs["payload_sha256"] == want["payload_sha256"]
                    and obs["frame_indices"] == want["frame_indices"]
                )
    write_lines(args.out / "planned_frame_inputs.jsonl", frames)
    write_lines(args.out / "planned_slots.jsonl", planned)
    dep.update(
        {
            name: hash_file(args.out / name)
            for name in ("planned_frame_inputs.jsonl", "planned_slots.jsonl")
        }
    )
    (args.out / "input_lock.json").write_text(json.dumps(dep, indent=2) + "\n")
    print(
        "Verified 120 distinct inputs and 240 planned slots", dep["planned_slots.jsonl"]
    )


if __name__ == "__main__":
    main()
