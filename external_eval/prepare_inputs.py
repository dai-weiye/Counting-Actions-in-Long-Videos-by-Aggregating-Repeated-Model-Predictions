"""Prepare label-free evaluation inputs and fetch only the 60 released test videos."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
REV = "4a9826be2bc6c86f9cab9222a06a55c224098958"
BASE = f"https://huggingface.co/datasets/sy1998/MLVU_Test/resolve/{REV}"
ANNOTATION = "mlvu_test/test-00000-of-00001.parquet"
OPTION = re.compile(r"^\(([A-F])\)\s*(\d+)\s*$", re.MULTILINE)


def get(url, **kwargs):
    for attempt in range(7):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, timeout=(10, 35), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            print(
                f"Read retry {attempt + 1}: {type(exc).__name__} ({url.split('?', 1)[0].rsplit('/', 1)[-1]})",
                flush=True,
            )
            if attempt == 6:
                raise
            time.sleep(2 ** min(attempt, 4))


def byte_range(name, start, end):
    response = get(
        f"{BASE}/{name}?segment={start}-{end}",
        headers={"Range": f"bytes={start}-{end}"},
    )
    received = response.headers.get("Content-Range", "")
    if (
        not received.startswith(f"bytes {start}-{end}/")
        or len(response.content) != end - start + 1
    ):
        raise ValueError(f"Inexact range for {name} at {start}")
    return response.content


class RemoteArchive(io.RawIOBase):
    """A seekable range reader for zipfile's small central-directory reads."""

    def __init__(self, name, size):
        self.name, self.size, self.pos = name, size, 0

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        pos = [offset, self.pos + offset, self.size + offset][whence]
        self.pos = max(0, pos)
        return self.pos

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos + size)
        if self.pos >= end:
            return b""
        data = byte_range(self.name, self.pos, end - 1)
        self.pos = end
        return data


def save(path, rows):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n")
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(cache, dev):
    cache.mkdir(parents=True, exist_ok=True)
    data = get(f"{BASE}/{ANNOTATION}").content
    (cache / "annotations.parquet").write_bytes(data)
    rows = pq.read_table(
        io.BytesIO(data),
        columns=["video_name", "question", "task_type", "question_id", "duration"],
    ).to_pylist()
    with dev.open() as stream:
        fit_rows = sorted(csv.DictReader(stream), key=lambda r: r["item_id"] + ".mp4")
    assert len(fit_rows) == 206
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(
        [[float(r["duration_s"])] for r in fit_rows], [int(r["gold"]) for r in fit_rows]
    )
    inputs = []
    for row in rows:
        if row["task_type"] != "count":
            continue
        matches = list(OPTION.finditer(row["question"]))
        assert [m[1] for m in matches] == list("ABCDEF")
        numerics = [m[2] for m in matches]
        assert len(set(numerics)) == 6
        stem = row["question"][: matches[0].start()].strip()
        name = row["video_name"]
        assert Path(name).name == name and name.startswith("test_count_")
        duration = float(row["duration"])
        inputs.append(
            {
                "id": row["question_id"],
                "video": name,
                "question": stem,
                "candidates": numerics,
                "duration": duration,
                "dev_duration_prediction": int(tree.predict([[duration]])[0]),
            }
        )
    inputs.sort(key=lambda r: int(r["id"].split("_")[1]))
    assert (
        len(inputs)
        == len({r["video"] for r in inputs})
        == len({r["id"] for r in inputs})
        == 60
    )
    save(ROOT / "questions.json", inputs)
    files = get(
        f"https://huggingface.co/api/datasets/sy1998/MLVU_Test/tree/{REV}"
    ).json()
    required = {r["video"] for r in inputs}
    result = {}
    for f in files:
        name = f["path"]
        if not name.endswith(".zip"):
            continue
        with zipfile.ZipFile(RemoteArchive(name, f["size"])) as z:
            for info in z.infolist():
                base = Path(info.filename).name
                if base not in required or "__MACOSX" in info.filename:
                    continue
                if base in result:
                    raise ValueError(f"Duplicate member for {base}")
                result[base] = {
                    "archive": name,
                    "archive_sha256": f["lfs"]["oid"],
                    "path": info.filename,
                    "header_offset": info.header_offset,
                    "compression": info.compress_type,
                    "compressed": info.compress_size,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                }
    if set(result) != required:
        raise ValueError(f"Missing test videos {required - result.keys()}")
    save(
        ROOT / "video_sources.json",
        {
            "repository": "sy1998/MLVU_Test",
            "revision": REV,
            "annotation_file": ANNOTATION,
            "annotation_sha256": digest(data),
            "videos": result,
        },
    )
    print(
        f"60 indexed videos, {sum(r['compressed'] for r in result.values()):,} compressed bytes",
        flush=True,
    )


def download_one(cache, item, parts):
    name, record = item
    dest = cache / name
    if dest.exists():
        crc = 0
        with dest.open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                crc = zlib.crc32(chunk, crc)
        if dest.stat().st_size == record["bytes"] and crc == record["crc32"]:
            return
        raise ValueError(f"Cached {name} does not match release")
    start = record["header_offset"]
    source = record["archive"]
    header = byte_range(source, start, start + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Bad ZIP local header")
    start += 30 + sum(struct.unpack("<HH", header[-4:]))
    csize = record["compressed"]
    partial = (cache / name).with_suffix(".zpart")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > csize:
        raise ValueError("Compressed partial too long")
    # Stage disjoint ranges on disk and append in order, preserving a contiguous,
    # resumable prefix if a transfer is interrupted.
    piece_size = 1024**2
    positions = list(range(offset, csize, piece_size))

    def pull(pos):
        length = min(piece_size, csize - pos)
        piece = cache / f"{name}.{pos}.{length}.piece"
        if not piece.exists() or piece.stat().st_size != length:
            chunk = byte_range(source, start + pos, start + pos + length - 1)
            pending = piece.with_suffix(".writing")
            pending.write_bytes(chunk)
            os.replace(pending, piece)
        return piece

    with partial.open("ab") as stream, ThreadPoolExecutor(parts) as workers:
        for piece in workers.map(pull, positions):
            stream.write(piece.read_bytes())
            stream.flush()
            piece.unlink()
    algo = record["compression"]
    if algo not in (0, 8):
        raise ValueError(f"Unsupported compression {algo}")
    decoder = zlib.decompressobj(-15) if algo == 8 else None
    pending = dest.with_suffix(".decode")
    crc = 0
    with partial.open("rb") as inp, pending.open("wb") as out:
        while chunk := inp.read(4 * 1024**2):
            data = decoder.decompress(chunk) if decoder else chunk
            out.write(data)
            crc = zlib.crc32(data, crc)
        if decoder:
            remain = decoder.flush()
            out.write(remain)
            crc = zlib.crc32(remain, crc)
            assert decoder.eof
    if pending.stat().st_size != record["bytes"] or crc != record["crc32"]:
        raise ValueError(f"{name}: ZIP member checksum mismatch")
    os.replace(pending, dest)
    partial.unlink()
    print(name, record["bytes"], "verified", flush=True)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--cache", type=Path, required=True)
    package = (
        ROOT.parent
        if (ROOT.parent / "data/action_count.csv").exists()
        else ROOT.parents[1] / "paper/ICME2027_REPRODUCIBILITY"
    )
    args.add_argument("--dev", type=Path, default=package / "data/action_count.csv")
    args.add_argument("--inventory", action="store_true")
    args.add_argument("--workers", type=int, default=6)
    args.add_argument("--parts", type=int, default=4)
    opts = args.parse_args()
    if opts.inventory or not (ROOT / "video_sources.json").exists():
        inventory(opts.cache, opts.dev)
    sources = json.loads((ROOT / "video_sources.json").read_text())["videos"]
    questions = json.loads((ROOT / "questions.json").read_text())
    with ThreadPoolExecutor(opts.workers) as executor:
        list(
            executor.map(
                lambda row: download_one(
                    opts.cache, (row["video"], sources[row["video"]]), opts.parts
                ),
                questions,
            )
        )


if __name__ == "__main__":
    main()
