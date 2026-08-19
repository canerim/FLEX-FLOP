"""Any script that times CUDA work must pin the device first.

torch.cuda.Event is created on the process's CURRENT device, not on the device
the tensors live on. Recording a pair of events on cuda:0 around work running on
cuda:7 does not raise -- it returns numbers, and they are wrong. Every wall-clock
figure in this project was measured that way before it was noticed, and it
survived because the long timings looked plausible: a full decode came out 29%
high while a sub-millisecond kernel came out a factor of several hundred low.

A source check rather than a runtime one, because reproducing the bug needs two
GPUs and the failure is silent on one.
"""
import re
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))
# The call, not the name. make_report.py now *describes* the bug in prose, and
# a bare-name match flagged it for not calling set_device -- which it has no
# reason to, since it times nothing.
USES_EVENT = [p for p in SCRIPTS
              if re.search(r"torch\.cuda\.Event\s*\(", p.read_text())]


def test_some_script_actually_times_cuda():
    """Guard the guard: if the pattern stops matching, the test is vacuous."""
    assert USES_EVENT, "no script uses torch.cuda.Event -- has the API changed?"


@pytest.mark.parametrize("path", USES_EVENT, ids=lambda p: p.name)
def test_event_timing_pins_the_device(path):
    src = path.read_text()
    assert "torch.cuda.set_device" in src, (
        f"{path.name} times CUDA work with torch.cuda.Event but never calls "
        f"torch.cuda.set_device. On any device other than cuda:0 the numbers "
        f"it prints are wrong and it will not tell you."
    )


@pytest.mark.parametrize(
    "path",
    [p for p in SCRIPTS
     if re.search(r"(?<![\"'])torch\.cuda\.synchronize\(\)", p.read_text())
     and "torch.cuda.Event(" in p.read_text()],
    ids=lambda p: p.name)
def test_bare_synchronize_pins_the_device(path):
    """torch.cuda.synchronize() with no argument syncs the current device."""
    src = path.read_text()
    assert "torch.cuda.set_device" in src, (
        f"{path.name} calls torch.cuda.synchronize() with no device. It syncs "
        f"the CURRENT device, so on any other device it returns before the work "
        f"is done and whatever it measures is not the work."
    )
