"""Construct the frozen seed study from public annotations and source videos."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def make_design(rows, video_dirs, recovery=False):
    import cv2
    rows = sorted(rows, key=lambda r: r["video"])
    if len(rows) != 206 or len({r["video"] for r in rows}) != len(rows):
        raise ValueError("Expected the 206 distinct MLVU Action Count development questions")
    holdout = [rows[i]["video"] for i in np.random.default_rng(0).permutation(206)[103:]]
    def rank(name):
        return hashlib.sha256(("MLVU seed replication | " + name).encode()).digest()
    names = sorted(holdout, key=rank)[:32]
    if recovery:
        names = [f"count_{i}.mp4" for i in (235, 216, 229, 201, 226, 232)]
    positions = []
    for seed in np.random.SeedSequence(20261124).spawn(24):
        rng = np.random.Generator(np.random.PCG64(seed))
        positions.append(sorted(map(int, rng.choice(256, 128, replace=False))))
    if len(set(map(tuple, positions))) != 24:
        raise ValueError("Repeated frame selection across streams")
    if recovery:
        positions = positions[:4]
    counts = {}
    checksums = {}
    slots = []
    for name in names:
        video = next((p/name for p in video_dirs if (p/name).is_file()), None)
        if video is None:
            raise FileNotFoundError(name)
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n < 256: raise ValueError(f"{name}: short or invalid video")
        counts[name] = n
        with video.open("rb") as source:
            checksums[name] = hashlib.file_digest(source, "sha256").hexdigest()
        pool = np.linspace(0, n-1, 256).astype(int)
        for s, mask in enumerate(positions):
            slots.append({"item_id": Path(name).stem, "group": s//4+1, "call": s%4+1,
                          "frame_indices": pool[mask].tolist()})
    order = np.random.default_rng(20261125).permutation(len(slots))
    return {"stream_positions": positions, "filenames": names, "source_frames": counts,
            "source_sha256": checksums, "slots": [slots[i] for i in order]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recovery", action="store_true", help="historical six videos with group one")
    a = p.parse_args()
    design = make_design(json.loads(a.annotations.read_text()), a.video_dir, a.recovery)
    text = json.dumps(design, separators=(",",":")) + "\n"
    if a.out.exists() and a.out.read_text() != text:
        raise SystemExit("Existing design does not match the reconstructed study")
    a.out.write_text(text)
    print(f"Saved {len(design['slots'])} frozen input slots")
