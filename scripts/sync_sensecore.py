# -*- coding: utf-8 -*-
"""RAMS 同步脚本：本地 ↔ 算力机（sensecore）双向同步

用途：保证本地与算力机代码/文档/探索进度一致。

同步规则（"产出双同步"）：
  - 方向一（本地 → 算力机）：代码、脚本、文档 —— 改动后即同步
  - 方向二（算力机 → 本地）：实验结果、产出报告 —— 生成后即同步

用法：
  python scripts/sync_sensecore.py --push    # 本地改动 → 算力机
  python scripts/sync_sensecore.py --pull    # 算力机产出 → 本地
  python scripts/sync_sensecore.py --both    # 双向（默认）
  python scripts/sync_sensecore.py --check   # 仅检查差异，不传输

同步内容：
  - 本地 → 算力机: rams/ scripts/ docs/ exp/ tests/ (排除 __pycache__/.pyc)
  - 算力机 → 本地: 同上目录下的新产出
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
REMOTE = "sensecore"
REMOTE_PROJ = "/data/RAMS/proj"

# 同步的目录（相对项目根），排除缓存
SYNC_DIRS = ["rams", "scripts", "docs", "exp", "tests", "configs"]
EXCLUDE = ["--exclude", "__pycache__", "--exclude", "*.pyc", "--exclude", "*.log",
           "--exclude", ".pytest_cache", "--exclude", "*.json"]


def run(cmd: list[str], check: bool = True, capture: bool = False) -> str:
    print(f"  $ {' '.join(cmd)[:120]}" + ("..." if len(" ".join(cmd)) > 120 else ""), flush=True)
    if capture:
        r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True)
        if check and r.returncode != 0:
            sys.exit(f"命令失败: {' '.join(cmd)}\n{r.stderr}")
        return r.stdout
    r = subprocess.run(cmd, cwd=str(PROJ))
    if check and r.returncode != 0:
        sys.exit(f"命令失败: {' '.join(cmd)}")
    return ""


def _tar_excludes() -> str:
    """tar 排除缓存文件的参数。"""
    return "--exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='.pytest_cache'"


def push() -> None:
    """本地 → 算力机（代码/脚本/文档改动即同步）"""
    print("=== 同步 本地 → 算力机 ===", flush=True)
    for d in SYNC_DIRS:
        src = PROJ / d
        if not src.exists():
            continue
        # tar 打包 → ssh 管道解包（绕过 Windows scp 无 --exclude 的限制）
        # 注意：Windows 路径要转正斜杠，否则 tar 会解析失败
        proj_posix = str(PROJ).replace("\\", "/")
        tar_cmd = f"tar -C {proj_posix} {_tar_excludes()} -cf - {d} | ssh -o BatchMode=yes {REMOTE} 'mkdir -p {REMOTE_PROJ} && tar -C {REMOTE_PROJ} -xf -'"
        print(f"  $ {tar_cmd[:120]}...", flush=True)
        r = subprocess.run(f"bash -c \"{tar_cmd}\"", shell=True)
        if r.returncode != 0:
            sys.exit(f"同步 {d} 失败")


def pull() -> None:
    """算力机 → 本地（实验结果/报告生成即同步）"""
    print("=== 同步 算力机 → 本地 ===", flush=True)
    proj_posix = str(PROJ).replace("\\", "/")
    for d in SYNC_DIRS:
        # 算力机 tar 打包 → 本地解包
        tar_cmd = f"ssh -o BatchMode=yes {REMOTE} 'cd {REMOTE_PROJ} && tar {_tar_excludes()} -cf - {d}' | tar -C {proj_posix} -xf -"
        print(f"  $ {tar_cmd[:120]}...", flush=True)
        r = subprocess.run(f"bash -c \"{tar_cmd}\"", shell=True)
        if r.returncode != 0:
            sys.exit(f"同步 {d} 失败")


def check_diff() -> None:
    """检查两边差异（列出本地有而算力机没有的文件）"""
    print("=== 检查差异 ===", flush=True)
    for d in SYNC_DIRS:
        src = PROJ / d
        if not src.exists():
            continue
        # 用 find 列出本地文件，和算力机对比
        stdout = run(["ssh", "-o", "BatchMode=yes", REMOTE,
                      f"ls {REMOTE_PROJ}/{d} 2>/dev/null"], check=False, capture=True)
        remote_files = set(stdout.split())
        local_files = {p.name for p in src.rglob("*") if p.is_file() and "__pycache__" not in str(p) and not str(p).endswith((".pyc", ".log"))}
        missing = local_files - remote_files
        if missing:
            print(f"  [{d}] 本地有、算力机缺: {sorted(missing)[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAMS 本地↔算力机同步")
    parser.add_argument("--push", action="store_true", help="本地→算力机")
    parser.add_argument("--pull", action="store_true", help="算力机→本地")
    parser.add_argument("--both", action="store_true", help="双向（默认）")
    parser.add_argument("--check", action="store_true", help="仅检查差异")
    args = parser.parse_args()

    if args.check:
        check_diff()
    elif args.push:
        push()
    elif args.pull:
        pull()
    else:
        pull()
        push()


if __name__ == "__main__":
    main()
