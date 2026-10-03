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
from bitstream_benchmark import DEFAULT_EXTENSION


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
    parser.add_argument("--bitstream-cohort", action="store_true",
                        help="After matched synthesis, run nine predeclared Kodak bitstream cases")
    parser.add_argument("--bitstream-manifest", type=Path,
                        default=HERE/'results/bitstream_kodak3x3/manifest.json')
    parser.add_argument("--extension", type=Path,
                        default=DEFAULT_EXTENSION,
                        help="Pinned FUFREF2 entropy extension directory")
    parser.add_argument("--roundtrip-after", action="store_true",
                        help="Also time full image->bytes->image after the decoder test")
    args = parser.parse_args()
    if min(args.hours, args.stable_minutes, args.poll_seconds) <= 0:
        parser.error("time limits must be positive")
    if args.roundtrip_after and not args.bitstream_cohort:
        parser.error("roundtrip requires --bitstream-cohort")
    folder = HERE / "results"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "matched_idle_watcher.jsonl"
    deadline = time.monotonic() + args.hours * 3600
    candidate = None
    candidate_since = None
    matched_done = False
    bitstream_done = False
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
            phase = ('cohort' if not matched_done else
                     'bitstream' if not bitstream_done else 'roundtrip')
            emit(log, {"event": f"{phase}_start", "gpu": candidate})
            output = folder / f"{phase}_idle_gpu{candidate}.log"
            if matched_done:
                command = [sys.executable, str(HERE / 'run_bitstream_cohort.py'),
                           '--extension', str(args.extension),
                           '--manifest', str(args.bitstream_manifest),
                           '--gpu', str(candidate),
                           '--blocks', str(5 if bitstream_done else args.blocks),
                           '--out', str(folder/f'{phase}_idle_gpu{candidate}')]
                if bitstream_done:
                    command += ['--include-encoder']
            else:
                command = [sys.executable, str(HERE / "run_cohort.py"),
                           "--gpu", str(candidate), "--blocks", str(args.blocks),
                           "--matched-kernels", "--resume",
                           "--out", str(folder/f'matched_idle_gpu{candidate}_cases')]
            with output.open("a") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            emit(log, {"event": f"{phase}_end", "gpu": candidate,
                       "returncode": result.returncode, "log": str(output)})
            if result.returncode != 0:
                try:
                    still_free = candidate in unused_gpus()
                except (OSError, subprocess.CalledProcessError, ValueError):
                    still_free = True  # stop rather than risk repeated GPU work
                if still_free:
                    emit(log, {"event": "aborted_after_failure", "phase": phase,
                               "gpu": candidate, "log": str(output)})
                    return result.returncode
                # Another workload arrived while measuring. Wait for a new
                # five-minute idle window rather than interfering with it.
                candidate = None
                candidate_since = None
                continue
            if result.returncode == 0:
                if phase == 'roundtrip' or (phase == 'bitstream' and not args.roundtrip_after):
                    return 0
                if phase == 'cohort':
                    if not args.bitstream_cohort:
                        return 0
                    matched_done = True
                elif phase == 'bitstream':
                    bitstream_done = True
            candidate = None
            candidate_since = None
        time.sleep(args.poll_seconds)
    emit(log, {"event": "expired"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
