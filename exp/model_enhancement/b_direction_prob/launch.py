# -*- coding: utf-8 -*-
"""启动器：等待 GPU 显存释放后运行 B_direction_prob 全量实验。

背景：算力机 GPU 被主作业（hunyuan 渲染 75GB）+ 其他探索任务占用，显存不足。
本脚本每 60s 检查一次空闲显存，≥6GB 时启动全量运行；期间不抢占主作业。
用法：nohup python3 exp/model_enhancement/b_direction_prob/launch.py > launch.log 2>&1 &
"""
import subprocess
import time

FREE_NEED_GB = 15.0
CMD = [
    "python3", "exp/model_enhancement/b_direction_prob/run_direction_prob.py",
    "--epochs", "30", "--device", "cuda",
    "--out-json", "exp/model_enhancement/b_direction_prob/results.json",
]

LOG = "exp/model_enhancement/b_direction_prob/run_full.log"


def free_gb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[0]
    free_mi, tot_mi = (int(x.split()[0]) for x in line.split(","))
    return free_mi / 1024.0


def main():
    while True:
        try:
            f = free_gb()
        except Exception as e:
            print(f"[launch] nvidia-smi error: {e}", flush=True)
            f = 0.0
        print(f"[launch] free GPU mem = {f:.1f} GB (need {FREE_NEED_GB})", flush=True)
        if f >= FREE_NEED_GB:
            print("[launch] memory OK, launching full run", flush=True)
            with open(LOG, "w") as lf:
                r = subprocess.run(CMD, stdout=lf, stderr=subprocess.STDOUT)
            print(f"[launch] full run exit code = {r.returncode}", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
