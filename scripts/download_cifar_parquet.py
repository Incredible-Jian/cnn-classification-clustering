"""Download the official CIFAR-10/100 image data from the Hugging Face mirror.

The original Toronto server is often slow from mainland China.  The
``uoft-cs/cifar10`` and ``uoft-cs/cifar100`` repositories host byte-identical
copies of the official CIFAR images in parquet format.  This script downloads
the four parquet shards to ``data/raw`` and converts them into compact NPZ
arrays used by the experiments.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://hf-mirror.com/datasets/uoft-cs"
FILES = {
    "cifar10_train": f"{BASE}/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "cifar10_test": f"{BASE}/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    "cifar100_train": f"{BASE}/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    "cifar100_test": f"{BASE}/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
}


def download(name: str, url: str) -> Path:
    dst = RAW / f"{name}.parquet"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"[skip] {dst.name} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
        return dst
    tmp = dst.with_suffix(".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (10 << 20) < (1 << 20):
                print(f"  {name}: {done/1e6:.0f}/{total/1e6:.0f} MB "
                      f"({done/max(time.time()-t0,1e-6)/1e6:.2f} MB/s)", flush=True)
    tmp.replace(dst)
    print(f"[done] {dst.name} {dst.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    return dst


def to_npz(name: str, parquet: Path, out: Path) -> None:
    table = pq.read_table(parquet)
    print(f"[convert] {name}: {table.num_rows} rows, columns={table.column_names}", flush=True)
    img_col = table.column("img").to_pylist()
    label_col = table.column("fine_label" if "fine_label" in table.column_names else "label").to_pylist()
    n = len(img_col)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    for i, item in enumerate(img_col):
        raw = item["bytes"] if isinstance(item, dict) else item
        images[i] = np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        if (i + 1) % 10000 == 0:
            print(f"  decoded {i+1}/{n}", flush=True)
    labels = np.asarray(label_col, dtype=np.int64)
    np.savez_compressed(out, images=images, labels=labels)
    print(f"[saved] {out.name} images={images.shape} labels={labels.shape}", flush=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {name: ex.submit(download, name, url) for name, url in FILES.items()}
        for name, fut in futs.items():
            paths[name] = fut.result()
    for ds in ("cifar10", "cifar100"):
        for split in ("train", "test"):
            key = f"{ds}_{split}"
            to_npz(key, paths[key], RAW / f"{key}.npz")
    meta = {
        "source": "https://huggingface.co/datasets/uoft-cs/cifar10 and /cifar100 (official CIFAR images)",
        "mirror": "https://hf-mirror.com",
        "files": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    (RAW / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
