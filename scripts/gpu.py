"""Which card to run on, asked of the driver rather than written down.

Forty-five scripts in this repository carried a hard-coded --device default,
and eight of them said cuda:2. GPU2 was ours when those defaults were written
and it is not ours now: running any of those eight today would put a process
on another user's card on a shared machine. The defaults could not notice,
because a number in a file cannot.

pick() applies the same policy as scripts/pick_gpu.sh -- a card this account
has to itself, most free memory first -- and never returns a card somebody
else is on. An explicit --device from the caller still wins; this only decides
what happens when nobody said.

    from gpu import pick
    ap.add_argument("--device", default=pick())
"""
from __future__ import annotations

import os
import subprocess

_CACHE: dict[str, str] = {}


def _owner_map() -> dict[str, set[str]]:
    """uuid -> the users with a process on that card. One driver query, one ps.

    An earlier version asked the driver once per card and ran ps once per
    process, which is eight subprocesses before any script could start; with
    forty-five scripts calling it that is the slowest import in the repository.
    """
    if "owners" in _CACHE:
        return _CACHE["owners"]
    out: dict[str, set[str]] = {}
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.splitlines()
    except Exception:
        _CACHE["owners"] = out
        return out
    pairs = []
    for line in rows:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[0].isdigit():
            pairs.append((p[0], p[1]))
    users = {}
    if pairs:
        try:
            ps = subprocess.run(
                ["ps", "-o", "pid=,user=", "-p", ",".join(p for p, _ in pairs)],
                capture_output=True, text=True, timeout=15).stdout
            for line in ps.splitlines():
                bits = line.split()
                if len(bits) >= 2:
                    users[bits[0]] = bits[1]
        except Exception:
            pass
    for pid, uuid in pairs:
        u = users.get(pid)
        if u:
            out.setdefault(uuid, set()).add(u)
    _CACHE["owners"] = out
    return out


def pick(fallback: str = "cuda:0") -> str:
    """A card this account has to itself, most free memory first.

    Falls back to the caller's own default only if the driver cannot be
    reached, so a machine without nvidia-smi behaves as it always did. If the
    driver answers and every card belongs to somebody else, that is worth
    failing on rather than borrowing one, so "cpu" is returned and the caller
    will be slow and obvious instead of quiet and rude.
    """
    if "device" in _CACHE:
        return _CACHE["device"]
    env = os.environ.get("FLEXUF_DEVICE")
    if env:
        _CACHE["device"] = env
        return env
    # Under CUDA_VISIBLE_DEVICES the driver's indices are not torch's. Asking
    # nvidia-smi and returning cuda:7 into a process that can see one card
    # raises "invalid device ordinal", which is what happened to
    # pipeline_stage_figs. Inside a restricted view there is nothing to pick.
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip() != "":
        _CACHE["device"] = "cuda:0"
        return "cuda:0"
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.splitlines()
    except Exception:
        _CACHE["device"] = fallback
        return fallback
    if not rows:
        _CACHE["device"] = fallback
        return fallback
    me = subprocess.run(["id", "-un"], capture_output=True,
                        text=True).stdout.strip()
    best, best_free = None, -1
    for r in rows:
        p = [x.strip() for x in r.split(",")]
        if len(p) < 4:
            continue
        idx, uuid = p[0], p[1]
        used = int("".join(c for c in p[2] if c.isdigit()) or 0)
        total = int("".join(c for c in p[3] if c.isdigit()) or 0)
        if _owner_map().get(uuid, set()) - {me}:
            continue
        if total - used > best_free:
            best, best_free = idx, total - used
    dev = f"cuda:{best}" if best is not None else "cpu"
    _CACHE["device"] = dev
    return dev


if __name__ == "__main__":
    print(pick())
