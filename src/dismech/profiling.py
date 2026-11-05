"""Lightweight phase timing for cost reporting.

Disabled unless the environment variable ``DISMECH_TIMING`` is set to
something other than "" or "0", so the hot loops pay only one module-global
boolean test per instrumented region when profiling is off.

Phases recorded (see ``benchmarks/faster_implementation/bench_cost_table.py``):

  ``broad``     neighbour search and culling of candidate stencil pairs
  ``narrow``    the distance query on surviving candidates
  ``assembly``  scatter of per-pair contact force and Jacobian into F and J
  ``solve``     the linear solve of the Newton system

Counters recorded:

  ``steps``            time steps taken
  ``newton_iters``     total Newton iterations
  ``newton_max``       largest Newton-iteration count of any single step
  ``backtracks``       line-search backtrackings (0 when use_line_search=False)
  ``pairs_candidate``  candidate stencil pairs entering the narrow phase
  ``pairs_active``     pairs that produced a force
  ``contact_calls``    contact-force evaluations (== Newton iterations)

Usage:

    DISMECH_TIMING=1 python my_simulation.py
    # then, from the driver:
    from dismech import profiling
    profiling.reset()
    ...run...
    profiling.dump("cost.json", extra={"n_nodes": 100})
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

ENABLED = os.environ.get("DISMECH_TIMING", "") not in ("", "0")

_times: dict[str, float] = defaultdict(float)
_counts: dict[str, float] = defaultdict(float)


class phase:
    """Context manager accumulating wall time into a named phase."""

    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if ENABLED:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if ENABLED:
            _times[self.name] += time.perf_counter() - self._t0
        return False


def count(name: str, value: float = 1.0) -> None:
    """Add to a named counter."""
    if ENABLED:
        _counts[name] += value


def observe_max(name: str, value: float) -> None:
    """Track the maximum value seen for a named counter."""
    if ENABLED and value > _counts[name]:
        _counts[name] = value


def reset() -> None:
    _times.clear()
    _counts.clear()


def snapshot() -> dict:
    return {"times_s": dict(_times), "counts": dict(_counts)}


def dump(path, extra: dict | None = None) -> dict:
    """Write the snapshot (plus any ``extra`` metadata) as JSON."""
    payload = snapshot()
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


if ENABLED:
    # Write the snapshot on interpreter exit when DISMECH_TIMING_OUT names a
    # file, so any simulation script can be profiled without editing it.
    _out = os.environ.get("DISMECH_TIMING_OUT", "")
    if _out:
        import atexit

        def _dump_at_exit():
            try:
                dump(_out)
            except Exception:      # never let profiling break a run
                pass

        atexit.register(_dump_at_exit)
