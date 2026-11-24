"""Verify all 240 receipts, join released answers and export only public, reproducible fields."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SLOTS = ("U1", "U2", "U3", "C4")


def prepare(qpath, source_path, parquet, run_dir, out, videos, dev_digests):
    questions = json.loads(qpath.read_text())
    sources = json.loads(source_path.read_text())
    assert len(questions) == 60 and len({q["id"] for q in questions}) == 60
    question_by_item = {Path(q["video"]).stem: q for q in questions}
    calls = []
    frame_records = []
    fingerprints = set()
    distinct_frames = {}
    http = Counter()
    finish = Counter()
    served = Counter()
    dates = set()
    for slot in SLOTS:
        file = run_dir / slot / "fourth_calls.jsonl"
        raw = [json.loads(line) for line in file.read_text().splitlines()]
        if len(raw) != 60 or {r["item_id"] for r in raw} != set(question_by_item):
            raise ValueError(
                "All 60 model responses in each slot must close before reading test answers"
            )
        for record in raw:
            item = record["item_id"]
            q = question_by_item[item]
            arm = "change" if slot == "C4" else "uniform"
            assert record["arm"] == arm
            count = record.get("count")
            status = record["status"]
            final = (record.get("final") or "").strip()
            if status == "ok":
                assert str(count) == final and final in q["candidates"]
            elif status == "blank_final":
                assert count is None and not final
            elif status == "invalid_final":
                assert count is None and final not in q["candidates"]
            else:
                assert count is None
            data = {
                "id": q["id"],
                "slot": slot,
                "status": status,
                "count": count,
                "finish_reason": record.get("finish_reason"),
            }
            calls.append(data)
            sel = record["frame_indices"]
            digest = record["payload_sha256"]
            assert sorted(set(sel)) == sel and len(sel) == 128 and len(digest) == 64
            if (item, arm) in distinct_frames:
                assert distinct_frames[item, arm] == (sel, digest), (
                    f"Three uniform inputs differ on {item}"
                )
            else:
                distinct_frames[item, arm] = sel, digest
                frame_records.append(
                    {
                        "id": q["id"],
                        "arm": arm,
                        "frame_indices": sel,
                        "payload_sha256": digest,
                    }
                )
            if record.get("fingerprint"):
                fingerprints.add(record["fingerprint"])
            if record.get("model_served"):
                served[record["model_served"]] += 1
            http[record.get("http_status")] += 1
            finish[record.get("finish_reason")] += 1
            dates.add(record["finished_utc"][:10])
    if len(calls) != 240:
        raise ValueError("Incomplete test run")
    assert len(distinct_frames) == 120
    for item in question_by_item:
        assert distinct_frames[item, "uniform"][1] != distinct_frames[item, "change"][1]
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != sources["annotation_sha256"]:
        raise ValueError("Annotation checksum changed")
    dev_rows = json.loads(dev_digests.read_text())
    assert len(dev_rows) == 206
    dev_sha = {r["sha256"] for r in dev_rows}
    dev_names = {r["video"] for r in dev_rows}
    video_provenance = []
    for q in questions:
        p = videos / q["video"]
        expected = sources["videos"][p.name]
        assert p.stat().st_size == expected["bytes"] and p.name not in dev_names
        with p.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        assert checksum not in dev_sha
        video_provenance.append(
            {
                "id": q["id"],
                "video": p.name,
                "bytes": expected["bytes"],
                "sha256": checksum,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_video_digests.json").write_text(
        json.dumps(video_provenance, indent=2) + "\n"
    )
    answers = {
        r["question_id"]: r["answer"]
        for r in pq.read_table(
            parquet, columns=["question_id", "task_type", "answer"]
        ).to_pylist()
        if r["task_type"] == "count"
    }
    assert set(answers) == {q["id"] for q in questions}
    items = []
    for q in questions:
        letter = answers[q["id"]].strip()
        if letter not in list("ABCDEF"):
            raise ValueError(f"Unknown official answer representation: {letter!r}")
        options = list(map(int, q["candidates"]))
        items.append(
            {
                "id": q["id"],
                "video": q["video"],
                "options": options,
                "gold": options[ord(letter) - ord("A")],
                "duration": q["duration"],
                "dev_duration_prediction": q["dev_duration_prediction"],
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=2) + "\n")
    for name, records in (
        ("calls.jsonl", calls),
        ("selected_frames.jsonl", frame_records),
    ):
        (out / name).write_text(
            "".join(
                json.dumps(r, separators=(",", ":")) + "\n"
                for r in sorted(
                    records,
                    key=lambda r: (
                        int(r["id"].split("_")[1]),
                        r.get("slot", r.get("arm")),
                    ),
                )
            )
        )
    stats = {
        "dates_utc": sorted(dates),
        "served_model_counts": served,
        "distinct_nonempty_service_fingerprints": len(fingerprints),
        "response_http_status": http,
        "finish_reason": finish,
        "protocol_sha256": hashlib.sha256(
            (ROOT / "protocol.md").read_bytes()
        ).hexdigest(),
        "input_annotations_sha256": sources["annotation_sha256"],
        "frame_inputs": len(frame_records),
        "overlap_check": {
            "development_videos": len(dev_rows),
            "test_videos": len(video_provenance),
            "shared_filenames": 0,
            "shared_file_sha256": 0,
        },
    }
    (out / "run_settings.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path, default=ROOT / "questions.json")
    p.add_argument("--sources", type=Path, default=ROOT / "video_sources.json")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--dev-digests", type=Path, default=ROOT / "dev_video_digests.json")
    p.add_argument("--out", type=Path, default=ROOT)
    a = p.parse_args()
    prepare(
        a.questions, a.sources, a.annotations, a.run_dir, a.out, a.videos, a.dev_digests
    )
