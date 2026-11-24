"""Fetch the pinned public checkpoint and verify each file against the release."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

REPO = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
REV = "01af461cdb9574acc09084a0ef94e216e142b085"
ROOT = Path(__file__).resolve().parent


def verify(path, meta):
    count = path.stat().st_size
    if count != meta["bytes"]:
        raise ValueError(f"Unexpected size for {path.name}")
    h = hashlib.sha256() if meta.get("sha256") else hashlib.sha1(f"blob {count}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != (meta.get("sha256") or meta["git_blob_sha1"]):
        raise ValueError(f"Checksum mismatch for {path.name}")


def fetch_file(cache, name, data):
    path = cache/name
    if path.exists():
        verify(path, data)
        return
    part = path.with_suffix(path.suffix+".part")
    for attempt in range(12):
        start = part.stat().st_size if part.exists() else 0
        if start == data["bytes"]:
            verify(part, data)
            part.replace(path)
            print(name, "verified", flush=True)
            return
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true&part={start}"
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-"} if start else {}, stream=True, timeout=(20,60)) as response:
                response.raise_for_status()
                if start and not (response.status_code==206 and response.headers.get("content-range","").startswith(f"bytes {start}-")):
                    raise ValueError("Server did not honor resume position")
                print(name,"downloading" if not start else f"resuming from {start}",flush=True)
                with part.open("ab" if start else "wb") as out:
                    for block in response.iter_content(1 << 20):
                        out.write(block)
            verify(part,data)
            part.replace(path)
            print(name, "verified", flush=True)
            return
        except (requests.RequestException, ValueError) as exc:
            print(name,type(exc).__name__,str(exc).split(" for url: ")[0][:125],flush=True)
            time.sleep(min(45,3*(attempt+1)))
    raise RuntimeError(f"Download failed for {name}")


def fetch_from_mirror(cache, name, data):
    path = cache/name
    if path.exists():
        verify(path, data)
        return
    part = path.with_suffix(path.suffix+".part")
    url = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo"
    session = requests.Session()
    session.trust_env = False
    size = data["bytes"]
    while (start := part.stat().st_size if part.exists() else 0) < size:
        end = min(size, start + (16 << 20)) - 1
        for attempt in range(5):
            try:
                query = {"Revision": "c8cba8079eb5d97f60347732c00e28a714026a7e",
                         "FilePath": name, "range_start": str(start)}
                resp = session.get(url, params=query, headers={"Range": f"bytes={start}-{end}"},
                                   timeout=(15, 160))
                resp.raise_for_status()
                if resp.headers.get("content-range") != f"bytes {start}-{end}/{size}" or len(resp.content)!=end-start+1:
                    raise ValueError("Inexact mirror range")
                with part.open("ab") as f:
                    f.write(resp.content)
                print(name,end+1,size,flush=True)
                break
            except (ValueError, requests.RequestException):
                if attempt == 4: raise
                time.sleep(3*(attempt+1))
    verify(part,data)
    part.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=ROOT/"model_files.json")
    parser.add_argument("--modelscope",action="store_true",help="Use the pinned, byte-verified ModelScope mirror")
    options=parser.parse_args()
    options.model_dir.mkdir(parents=True,exist_ok=True)
    if options.manifest.exists():
        manifest=json.loads(options.manifest.read_text())
        if manifest.get("revision")!=REV or manifest.get("repo_id")!=REPO:
            raise ValueError("Unexpected model manifest")
    else:
        resp=requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REV}",params={"blobs":"true"},timeout=(10,40))
        resp.raise_for_status()
        obj=resp.json()
        if obj["sha"]!=REV: raise ValueError("Revision mismatch")
        meta={}
        for f in obj["siblings"]:
            name=f["rfilename"]
            if name in {".gitattributes","README.md"}: continue
            if name!=Path(name).name:raise ValueError("Unexpected checkpoint filename")
            meta[name]={"bytes":f["size"],"git_blob_sha1":f["blobId"]}
            if f.get("lfs"):meta[name]["sha256"]=f["lfs"]["sha256"]
        manifest={"repo_id":REPO,"revision":REV,"files":meta}
        options.manifest.write_text(json.dumps(manifest,indent=2)+"\n")
    for name,meta in sorted(manifest["files"].items(),key=lambda pair:pair[1]["bytes"]):
        fetch = fetch_from_mirror if options.modelscope else fetch_file
        fetch(options.model_dir,name,meta)

if __name__=="__main__":main()
