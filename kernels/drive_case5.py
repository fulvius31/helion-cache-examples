"""In-process Case-5 driver: 5a A/B/B', 5b warm-start list(), 5c degrade + bonus.

Everything runs in ONE process so the exact-match cache keys stay stable (the
key embeds the default lambda's code-object address, which differs across
processes); ``kernel.reset()`` before each "machine" forces a fresh bind +
on-disk cache consult, and the cache class / effort are switched via the
kernel's own (mutable) Settings. Printed evidence on stderr, numbers in
RESULT_JSON.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import best_perf, apply_matmul_edit, restore_matmul  # noqa: E402

REMOTE = ROOT / "remote_store"
REMOTE_CORRUPT = ROOT / "remote_store_corrupt"


def snap(counters) -> dict:
    a = counters.get("autotune", {})
    return {"hit": a.get("cache_hit", 0), "miss": a.get("cache_miss", 0), "put": a.get("cache_put", 0)}


def n_remote(path: Path) -> int:
    return len(list(path.glob("*.json")))


def n_local(dirname: str) -> int:
    d = ROOT / "cache" / dirname
    return len(list(d.glob("*.best_config"))) if d.exists() else 0


def main() -> None:
    logging.getLogger("helion").setLevel(logging.DEBUG)
    from helion._utils import counters
    from helion._testing import DEVICE, HALF_DTYPE
    from kernels.matmul_demo import matmul

    size = 1024
    x = torch.randn([size, size], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([size, size], device=DEVICE, dtype=HALF_DTYPE)
    res: dict = {}

    def machine(label: str, cache_dir: str, *, edited: bool = False,
                effort: str = "quick", cache_cls: str = "RemoteAutotuneCache",
                backend: str | None = "demo_remote_backend.FileBackend",
                remote_dir: Path = REMOTE, assert_ref=True) -> dict:
        os.environ["HELION_CACHE_DIR"] = str(ROOT / "cache" / cache_dir)
        os.environ["HELION_AUTOTUNE_EFFORT"] = effort
        os.environ["HELION_AUTOTUNE_CACHE"] = cache_cls
        if backend is None:
            os.environ.pop("HELION_REMOTE_CACHE_BACKEND", None)
        else:
            os.environ["HELION_REMOTE_CACHE_BACKEND"] = backend
        os.environ["HELION_DEMO_REMOTE_DIR"] = str(remote_dir)
        matmul.settings.autotune_effort = effort
        matmul.settings.autotune_cache = cache_cls
        matmul.reset()
        before = snap(counters)
        local_before = n_local(cache_dir)
        out = matmul(x, y)
        after = snap(counters)
        delta = {k: after[k] - before[k] for k in before}
        if label in ("B", "B'"):
            assert delta == {"hit": 1, "miss": 0, "put": 0}, (label, delta)
        else:
            assert delta == {"hit": 0, "miss": 1, "put": 1}, (label, delta)
        if label == "B":
            assert local_before == 0 and n_local(cache_dir) == 1
        ref = torch.matmul(x, y) * (2.0 if edited else 1.0)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
        ms = best_perf(matmul, x, y, iters=30, warmup=5)
        return {"cache_dir": cache_dir, "perf_ms": ms, "local_entries": n_local(cache_dir),
                "remote_files": n_remote(remote_dir), "counters": delta,
                "environment": {k: os.environ[k] for k in (
                    "HELION_CACHE_DIR", "HELION_AUTOTUNE_EFFORT", "HELION_AUTOTUNE_CACHE",
                    "HELION_REMOTE_CACHE_BACKEND", "HELION_DEMO_REMOTE_DIR",
                ) if k in os.environ}}

    # ================== 5a: exact-match sharing ============================
    for p in list(REMOTE.glob("*.json")):
        p.unlink()
    res["A"] = machine("A", "case5_A", cache_cls="RemoteAutotuneCache", effort="quick")
    print(f"  [A] quick-tune: {res['A']}", flush=True)
    res["B"] = machine("B", "case5_B", cache_cls="RemoteAutotuneCache", effort="quick")
    print(f"  [B] remote read: {res['B']}", flush=True)
    res["Bp"] = machine("B'", "case5_B", cache_cls="LocalAutotuneCache", backend=None, effort="quick")
    print(f"  [B'] local no-backend: {res['Bp']}", flush=True)

    # ================== 5b: warm-start list() ==============================
    try:
        apply_matmul_edit()
        res["C"] = machine("C", "case5_C", edited=True, cache_cls="LocalAutotuneCache",
                           backend="demo_remote_backend.FileBackend", effort="quick")
        print(f"  [C] edited warm-start: {res['C']}", flush=True)
    finally:
        restore_matmul()

    # ================== 5c: graceful degradation ===========================
    res["D"] = machine("D", "case5_D", cache_cls="RemoteAutotuneCache",
                       backend="demo_remote_backend.FailingBackend", effort="full")
    print(f"  [D] full w/ FailingBackend: {res['D']}", flush=True)

    # ================== 5c bonus: corrupt remote ===========================
    REMOTE_CORRUPT.mkdir(parents=True, exist_ok=True)
    for p in list(REMOTE_CORRUPT.glob("*.json")):
        p.unlink()
    for src in REMOTE.glob("*.json"):
        (REMOTE_CORRUPT / src.name).write_text(src.read_text())
    (REMOTE_CORRUPT / "corrupt_zz.json").write_text('{"broken": ')
    try:
        apply_matmul_edit()
        res["Ccorrupt"] = machine("Ccorrupt", "case5_C_corrupt", edited=True,
                                  cache_cls="LocalAutotuneCache",
                                  backend="demo_remote_backend.FileBackend",
                                  remote_dir=REMOTE_CORRUPT, effort="quick")
        print(f"  [bonus] corrupt store: {res['Ccorrupt']}", flush=True)
    finally:
        restore_matmul()

    # record counter snapshots for the orchestrator (evidence greps the log)
    res["_counters"] = snap(counters)
    print("RESULT_JSON=" + json.dumps(res))
    print("RUN_OK")


if __name__ == "__main__":
    main()
