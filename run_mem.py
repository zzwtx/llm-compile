#!/usr/bin/env python3
"""
Small utility to monitor GPU memory of two GPUs while running build/convert commands.

It spawns a monitor thread that polls `nvidia-smi` for memory.used every `--interval` seconds
and records per-GPU min/max/peak values relative to a baseline sampled before running the
commands. Results are printed and written to `--out` (JSON).

Usage (defaults shown):
    python3 run_mem.py \
      --gpus 0 1 \
      --interval 0.5 \
      --out build_mem_results.json

The script will run two commands sequentially (the conversion and the trtllm-build) and monitor
GPU memory while they execute.

Be sure `nvidia-smi` is available and visible to the user running this script.
"""

import argparse
import json
import shlex
import subprocess
import threading
import time
import os
from collections import defaultdict
from datetime import datetime
from typing import List

# optional psutil usage for more accurate per-process RSS + descendants
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    psutil = None
    _HAVE_PSUTIL = False


def sample_gpus(gpu_indices: List[int]) -> List[int]:
    """Return list of memory.used in MiB for requested GPU indices in same order."""
    # Query all GPUs and parse lines. Use nounits/nocsv header for easy parsing.
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi query failed: {e}")

    # out lines like: "0, 1234\n1, 2048\n"
    mem_map = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            mem = int(parts[1])
        except ValueError:
            continue
        mem_map[idx] = mem

    return [mem_map.get(int(i), 0) for i in gpu_indices]


def get_system_mem_used_mib() -> int:
    """Return system used memory in MiB (MemTotal - MemAvailable) by parsing /proc/meminfo.

    This avoids a dependency on psutil and works on Linux.
    """
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                try:
                    meminfo[key] = int(val)
                except ValueError:
                    continue
    except FileNotFoundError:
        # Non-linux fallback: return 0
        return 0

    if "MemTotal" in meminfo and "MemAvailable" in meminfo:
        used_kib = meminfo["MemTotal"] - meminfo["MemAvailable"]
        return max(0, used_kib // 1024)
    return 0


def get_process_rss_mib(pid: int, include_descendants: bool = True) -> int:
    """Return RSS (resident set) in MiB for given pid.

    If psutil is available and include_descendants is True, will sum RSS of pid and all descendants.
    Fallback: read /proc/<pid>/status for VmRSS and do not include descendants.
    """
    if pid is None:
        return 0
    try:
        if _HAVE_PSUTIL and include_descendants:
            try:
                p = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return 0
            rss = p.memory_info().rss
            for ch in p.children(recursive=True):
                try:
                    rss += ch.memory_info().rss
                except Exception:
                    pass
            return max(0, rss // (1024 * 1024))
        else:
            # read own process VmRSS only
            stat_path = f"/proc/{pid}/status"
            with open(stat_path, "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            # value is in kB
                            try:
                                kib = int(parts[1])
                                return max(0, kib // 1024)
                            except ValueError:
                                return 0
            return 0
    except Exception:
        return 0


class MonitorThread(threading.Thread):
    def __init__(self, gpu_indices: List[int], interval: float = 0.5, tracked_pids: List[int] = None):
        super().__init__()
        self.gpu_indices = gpu_indices
        self.interval = interval
        # don't use attribute name `_stop` because Thread has an internal
        # attribute/method with the same name; use `_stop_event` instead.
        self._stop_event = threading.Event()
        # records lists of (t_rel, value). Keys: int gpu idx, and special keys '_cpu', '_total', 'pid:<pid>'
        self.data = defaultdict(list)
        self.start_time = None
        # optional list of PIDs to track (shared, updated by run_command)
        self.tracked_pids = tracked_pids if tracked_pids is not None else []

    def run(self):
        self.start_time = time.time()
        while not self._stop_event.is_set():
            t = time.time() - self.start_time
            try:
                mems = sample_gpus(self.gpu_indices)
            except Exception:
                mems = [None for _ in self.gpu_indices]

            # record per-gpu
            for idx, m in zip(self.gpu_indices, mems):
                self.data[int(idx)].append((t, int(m) if m is not None else None))

            # sample system CPU used memory
            try:
                cpu_used = get_system_mem_used_mib()
            except Exception:
                cpu_used = None
            self.data["_cpu"].append((t, int(cpu_used) if cpu_used is not None else None))

            # total = cpu_used + sum(gpu used)
            try:
                gpu_sum = sum(m for m in mems if m is not None)
                total = (cpu_used or 0) + (gpu_sum or 0)
            except Exception:
                total = None
            self.data["_total"].append((t, int(total) if total is not None else None))

            # sample tracked pids (RSS MiB), include descendants if possible
            for pid in list(self.tracked_pids):
                rss = get_process_rss_mib(int(pid), include_descendants=True)
                self.data[f"pid:{pid}"].append((t, int(rss)))

            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


def run_command(cmd_str: str, cwd: str = None) -> int:
    print(f"Running: {cmd_str}")
    # Use shell to allow complex commands; start process so its pid can be tracked
    process = subprocess.Popen(cmd_str, shell=True)
    print(f"Started process pid={process.pid} for command")
    ret = process.wait()
    print(f"Command finished with return code {ret}: {cmd_str}")
    return ret


def analyze(data: dict, baseline: List[int], gpu_indices: List[int]):
    # data: dict mapping gpu idx -> list of (t, mem)
    report = {}
    for i, gpu in enumerate(gpu_indices):
        records = data.get(int(gpu), [])
        mems = [m for (_, m) in records if m is not None]
        if not mems:
            report[int(gpu)] = {
                "baseline": baseline[i],
                "samples": 0,
                "min": None,
                "max": None,
                "peak": None,
                "delta_peak_vs_baseline": None,
                "delta_min_vs_baseline": None,
            }
            continue
        minv = min(mems)
        maxv = max(mems)
        peak = maxv
        report[int(gpu)] = {
            "baseline": baseline[i],
            "samples": len(mems),
            "min": minv,
            "max": maxv,
            "peak": peak,
            "delta_peak_vs_baseline": peak - baseline[i],
            "delta_min_vs_baseline": minv - baseline[i],
        }
    # add CPU and total aggregated stats if present
    cpu_records = data.get("_cpu", [])
    cpu_mems = [m for (_, m) in cpu_records if m is not None]
    if cpu_mems:
        report["_cpu"] = {"samples": len(cpu_mems), "min": min(cpu_mems), "max": max(cpu_mems)}
    else:
        report["_cpu"] = {"samples": 0, "min": None, "max": None}

    total_records = data.get("_total", [])
    total_mems = [m for (_, m) in total_records if m is not None]
    if total_mems:
        report["_total"] = {"samples": len(total_mems), "min": min(total_mems), "max": max(total_mems)}
    else:
        report["_total"] = {"samples": 0, "min": None, "max": None}

    # include per-pid summary
    pid_entries = {k: v for k, v in data.items() if isinstance(k, str) and k.startswith("pid:")}
    for pid_key, records in pid_entries.items():
        mems = [m for (_, m) in records if m is not None]
        if mems:
            report[pid_key] = {"samples": len(mems), "min": min(mems), "max": max(mems)}
        else:
            report[pid_key] = {"samples": 0, "min": None, "max": None}

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="GPU indices to monitor")
    parser.add_argument("--interval", type=float, default=0.5, help="sampling interval in seconds")
    parser.add_argument("--out", type=str, default="build_mem_results.json", help="output JSON file")
    parser.add_argument("--cwd", type=str, default=None, help="working directory for commands")
    parser.add_argument("--cmd", "-c", action="append", help="custom command to run under monitor; may be passed multiple times")
    args = parser.parse_args()

    gpu_indices = args.gpus
    interval = args.interval

    # sample baseline (a few quick samples to be robust)
    print("Sampling baseline GPU memory (3 samples)...")
    baseline_samples = []
    for _ in range(3):
        baseline_samples.append(sample_gpus(gpu_indices))
        time.sleep(0.1)
    # take the minimum across the small baseline window as baseline
    baseline = [min(samples[i] for samples in baseline_samples) for i in range(len(gpu_indices))]
    print(f"Baseline (MiB) per gpu {gpu_indices}: {baseline}")

    # tracked_pids is a shared list the MonitorThread will sample; run_command will append PIDs here
    tracked_pids = []
    monitor = MonitorThread(gpu_indices=gpu_indices, interval=interval, tracked_pids=tracked_pids)
    monitor.daemon = True
    monitor.start()

    # Commands to run (default pair). Users may supply --cmd multiple times to override.
    cmd1 = (
        "python3 /path/to/convert_checkpoint.py "
        "--model_dir=/path/to/llm/engine "
        "--output_dir=/path/to/llm/checkpoint "
        "--dtype float16"
    )

    cmd2 = (
        "trtllm-build --checkpoint_dir=/path/to/model"
        "--output_dir=/path/to/llm/engine"
        "--gemm_plugin=float16 "
        "--gpt_attention_plugin=float16 "
        "--max_batch_size=4 "
        "--max_input_len=2048 "
        "--max_seq_len=3072 "
        "--max_multimodal_len=1296"
    )

    # Build the list of commands to run. If user provided --cmd, use those (in order); otherwise use default pair.
    if args.cmd:
        commands_to_run = args.cmd
    else:
        commands_to_run = [cmd1, cmd2]

    start_ts = datetime.utcnow().isoformat() + "Z"
    cmd_results = []

    try:
        for i, cmd in enumerate(commands_to_run, start=1):
            print(f"Starting command {i}: ...")
            p = subprocess.Popen(cmd, shell=True, cwd=args.cwd)
            tracked_pids.append(p.pid)
            ret = p.wait()
            cmd_results.append({"cmd": cmd, "returncode": ret, "pid": p.pid})
            try:
                tracked_pids.remove(p.pid)
            except ValueError:
                pass

    finally:
        # stop monitor
        monitor.stop()
        # give monitor a moment to flush
        monitor.join(timeout=2.0)

    end_ts = datetime.utcnow().isoformat() + "Z"

    # analyze
    report = analyze(monitor.data, baseline, gpu_indices)

    out = {
        "start_time": start_ts,
        "end_time": end_ts,
        "gpu_indices": gpu_indices,
        "baseline": {str(g): b for g, b in zip(gpu_indices, baseline)},
        "per_gpu": report,
        "raw_samples": {str(g): monitor.data.get(int(g), []) for g in gpu_indices},
        "cpu_samples": monitor.data.get("_cpu", []),
        "total_samples": monitor.data.get("_total", []),
        "per_pid_samples": {k: v for k, v in monitor.data.items() if isinstance(k, str) and k.startswith("pid:")},
        "commands": cmd_results,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("Monitoring complete. Summary:")
    # print GPU stats first
    for g in gpu_indices:
        stats = report.get(int(g), {})
        print(f" GPU {g}: baseline={stats.get('baseline')} MiB, min={stats.get('min')} MiB, max={stats.get('max')} MiB, peak={stats.get('peak')} MiB, delta_peak={stats.get('delta_peak_vs_baseline')} MiB")

    # CPU and total
    cpu_stats = report.get("_cpu", {})
    print(f" System CPU used (MiB): samples={cpu_stats.get('samples')}, min={cpu_stats.get('min')}, max={cpu_stats.get('max')}")
    total_stats = report.get("_total", {})
    print(f" CPU+GPU total (MiB): samples={total_stats.get('samples')}, min={total_stats.get('min')}, max={total_stats.get('max')}")

    # per-pid
    for k, v in report.items():
        if isinstance(k, str) and k.startswith("pid:"):
            print(f" {k}: samples={v.get('samples')}, min={v.get('min')} MiB, max={v.get('max')} MiB")

    print(f"Full results saved to {args.out}")


if __name__ == '__main__':
    main()
