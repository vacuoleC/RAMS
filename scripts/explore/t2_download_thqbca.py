# -*- coding: utf-8 -*-
"""THQBCA 断点续传下载器（Zenodo 支持 range 请求）

用法：python scripts/explore/t2_download_thqbca.py --out data/public/thqbca
中断后重跑会自动从已有字节续传。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

URL = "https://zenodo.org/records/13917285/files/THQBCA-V2.rar"
CHUNK = 1 << 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public/thqbca")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "THQBCA-V2.rar"

    # 先 HEAD 拿总大小
    head = requests.head(URL, timeout=30, allow_redirects=True)
    total = int(head.headers.get("content-length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    print(f"目标: {total / 1e6:.1f} MB  已有: {have / 1e6:.1f} MB", flush=True)

    if have >= total:
        print("已完成，无需下载", flush=True)
        return

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(URL, stream=True, timeout=120, headers=headers,
                      allow_redirects=True) as r:
        mode = "ab" if have else "wb"
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    have += len(chunk)
                    if have % (50 << 20) < CHUNK:
                        print(f"  {have / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
    print(f"完成: {dest}  {dest.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
