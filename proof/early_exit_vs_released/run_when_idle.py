"""Run the matched-kernel cohort once a GPU stays completely unused.

This process only polls nvidia-smi while waiting. The benchmark itself checks
for competing GPU processes again at startup and during every paired block.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def emit(log: Path, event: dict) -> None:
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with log.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def unused_gpus() -> list[int]:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader"], text=True)
    used_uuids = {line.split(",")[0].strip() for line in apps.splitlines() if "," in line}
    free = []
    for line in gpu.splitlines():
        index, uuid, memory, utilization = [field.strip() for field in line.split(",")]
        if uuid not in used_uuids and int(memory) < 256 and int(utilization) < 5:
            free.append(int(index))
    return free


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--stable-minutes", type=float, default=5)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--blocks", type=int, default=20)
    args = parser.parse_args()
    if min(args.hours, args.stable_minutes, args.poll_seconds) <= 0:
        parser.error("time limits must be positive")
    folder = HERE / "results"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "matched_idle_watcher.jsonl"
    deadline = time.monotonic() + args.hours * 3600
    candidate = None
    candidate_since = None
    emit(log, {"event": "started", "hours": args.hours,
               "stable_minutes": args.stable_minutes})
    while time.monotonic() < deadline:
        try:
            free = unused_gpus()
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            emit(log, {"event": "poll_error", "error": str(error)})
            free = []
        now = time.monotonic()
        if candidate not in free:
            candidate = min(free) if free else None
            candidate_since = now if candidate is not None else None
            if candidate is not None:
                emit(log, {"event": "candidate", "gpu": candidate})
        if candidate is not None and now - candidate_since >= args.stable_minutes * 60:
            emit(log, {"event": "cohort_start", "gpu": candidate})
            output = folder / f"matched_idle_gpu{candidate}.log"
            command = [sys.executable, str(HERE / "run_cohort.py"),
                       "--gpu", str(candidate), "--blocks", str(args.blocks),
                       "--matched-kernels"]
            with output.open("a") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            emit(log, {"event": "cohort_end", "gpu": candidate,
                       "returncode": result.returncode, "log": str(output)})
            if result.returncode == 0:
                return 0
            candidate = None
            candidate_since = None
        time.sleep(args.poll_seconds)
    emit(log, {"event": "expired"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
