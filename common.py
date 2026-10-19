"""Shared helpers for the Helion autotune-cache demo suite.

Each case orchestrates several *runs*.  A run is a fresh subprocess (so every
run's environment — cache dir, effort, force flags, remote backend — is fully
auditable) whose stderr is captured verbatim to ``logs/<case>_<step>.log``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"
REMOTE_DIR = ROOT / "remote_store"

# Same values on every run: verbose helion DEBUG logging on stderr, and a
# non-LLM autotuner.  The default autotuner in this Helion snapshot is the
# LLM-seeded ``LFBOTreeSearch``, whose stage-1 makes a live LLM HTTP call
# that cannot run headlessly here.  ``LFBOPatternSearch`` is the classic
# search (base_search.py: best-available population + random padding) and it
# emits the population log lines the PASS checks grep for.  Documented in
# README.md.
BASE_ENV = {
    "HELION_LOGS": "+all",
    "HELION_AUTOTUNE_LOG_LEVEL": "DEBUG",  # surfaces the warm-start debug lines
    "HELION_AUTOTUNER": "LFBOPatternSearch",
    "CUDA_VISIBLE_DEVICES": os.environ.get(
        "DEMO_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ),
}


def python_bin() -> str:
    """Return the venv python when present, else the current interpreter."""
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def fresh_cache_dir(name: str) -> Path:
    """Return an isolated cache dir for *name*, removing any prior entries."""
    d = CACHE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*.best_config"):
        p.unlink()
    return d


class RunResult:
    """Outcome of one ``run_once`` subprocess."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed_s: float,
        output: str,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed_s = elapsed_s
        self.output = output  # stdout followed by stderr, not interleaved

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def result_json(self) -> dict[str, Any] | None:
        """Parse the driver's RESULT_JSON=... line from stdout, if present."""
        for line in self.stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line[len("RESULT_JSON=") :])
        return None

    def contains(self, text: str) -> bool:
        return text in self.output

    def perfs(self) -> float | None:
        rj = self.result_json()
        if rj is None:
            return None
        return float(rj["best_perf_ms"])

    def counters(self) -> dict[str, int]:
        rj = self.result_json()
        if rj is None:
            return {}
        return {k: int(v) for k, v in rj.get("counters", {}).items()}


def run_once(
    cmd: Sequence[str],
    env: Mapping[str, str] | None = None,
    log: str | None = None,
    *,
    cwd: Path | None = None,
    timeout: int | None = 3600,
) -> RunResult:
    """Run with explicit Helion settings, independent of ambient HELION_* flags.

    Keep the rest of the environment (including the selected CUDA device and
    PYTHONPATH). Write output and a command/environment/timing sidecar per run.
    """
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("HELION_")}
    full_env.update(BASE_ENV)
    if env:
        full_env.update(env)
    cwd = cwd or ROOT
    log_path = None
    if log:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"{log}.log"

    def save_log(output: str, elapsed: float, returncode: int) -> None:
        if log_path is None:
            return
        log_path.write_text(output)
        log_path.with_suffix(".meta.json").write_text(json.dumps({
            "command": list(cmd),
            "cwd": str(cwd),
            "environment": {k: v for k, v in full_env.items()
                            if k.startswith("HELION_") or k == "CUDA_VISIBLE_DEVICES"},
            "elapsed_s": elapsed,
            "returncode": returncode,
        }, indent=2) + "\n")

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        combined = f"{out}\n{err}\n[TIMEOUT after {int(elapsed)}s]"
        save_log(combined, elapsed, -124)
        return RunResult(-124, out, err, elapsed, combined)
    elapsed = time.perf_counter() - start

    combined = proc.stdout + "\n" + proc.stderr
    save_log(combined, elapsed, proc.returncode)
    return RunResult(proc.returncode, proc.stdout, proc.stderr, elapsed, combined)


def best_perf(
    fn: Any, *args: Any, iters: int = 50, warmup: int = 10, **kwargs: Any
) -> float:
    """Mean steady-state CUDA-event time per invocation, in milliseconds."""
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


def dump_cache_entries(cache_dir: Path) -> dict[str, Any]:
    """Return {filename: parsed JSON} for every ``*.best_config`` in *cache_dir*."""
    entries: dict[str, Any] = {}
    if not cache_dir.exists():
        return entries
    for p in sorted(cache_dir.glob("*.best_config")):
        try:
            entries[p.name] = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            entries[p.name] = {"__parse_error__": str(e)}
    return entries


def entry_keys(cache_dir: Path) -> list[str]:
    return sorted(p.name for p in cache_dir.glob("*.best_config"))


def check_evidence(result: RunResult, text: str, label: str) -> bool:
    """Record whether *text* appears in a run's output; used for PASS checks."""
    found = result.contains(text)
    print(f"  [{'OK' if found else 'MISSING'}] {label}: \"{text}\"")
    return found


# --- case-1 / case-5b in-place kernel-body edit ----------------------------
# The exact-match cache keys on the kernel function's OWN source
# (inspect.getsource(kernel.fn)), so a real in-body edit invalidates it while
# leaving the tiling skeleton + config space untouched (warm-start survives).

_MATMUL_DEMO = ROOT / "kernels" / "matmul_demo.py"
_ORIG_STORE = "        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))"
_EDIT_STORE = "        out[tile_m, tile_n] = (epilogue(acc, (tile_m, tile_n)) * 2.0).to(out.dtype)"


def matmul_edit_applied() -> bool:
    return _EDIT_STORE in _MATMUL_DEMO.read_text()


def apply_matmul_edit() -> None:
    """Edit the kernel body in place: multiply the store result by 2.0."""
    text = _MATMUL_DEMO.read_text()
    if _EDIT_STORE in text:
        return
    assert _ORIG_STORE in text, "expected store line not found; example drifted?"
    _MATMUL_DEMO.write_text(text.replace(_ORIG_STORE, _EDIT_STORE, 1))


def restore_matmul() -> None:
    """Undo the case-1 edit (idempotent, restores the CI-tested body)."""
    text = _MATMUL_DEMO.read_text()
    if _EDIT_STORE not in text:
        return
    _MATMUL_DEMO.write_text(text.replace(_EDIT_STORE, _ORIG_STORE, 1))


# --- case-4 poisoned-entry helpers ------------------------------------------

def _poison_value(knob: str, v: object) -> object:
    """Shrink block sizes (halve, min 16) and num_warps (halve, min 1)."""
    if knob == "num_warps" and isinstance(v, int):
        return max(1, v // 2)
    if isinstance(v, int) and "block" in knob:
        return max(16, v // 2)
    return v


def poison_entry(entry_path: Path) -> dict:
    """Rewrite a .best_config entry with a valid-but-slow config.

    Both halves of the entry are updated and kept consistent:
      * ``config``      -> sparse JSON config dict (used by the exact-hit path)
      * ``flat_config`` -> the canonical flattened vector (used to seed a
        warm-start population)
    The ``key`` fields are left untouched so the entry still *matches*.
    """
    data = json.loads(entry_path.read_text())
    poisoned = json.loads(json.dumps(data))  # deep copy
    cfg = json.loads(poisoned["config"])  # config is a JSON *string* on disk
    orig = dict(cfg)

    # Locate where each sparse config value (or element) sits in the canonical
    # flat_config vector, so poisoning both halves stays consistent.
    positions: dict[tuple[str, int], int] = {}
    used: set[int] = set()

    def match(key: tuple[str, int], want: object) -> None:
        if key in positions:
            return
        for i, fv in enumerate(flat):
            if fv == want and i not in used:
                positions[key] = i
                used.add(i)
                return

    if isinstance(poisoned.get("flat_config"), str):
        flat = json.loads(poisoned["flat_config"])
        for knob, val in orig.items():
            if isinstance(val, list):
                for j, ev in enumerate(val):
                    match((knob, j), ev)
            else:
                match((knob, 0), val)

        for knob, val in list(cfg.items()):
            if isinstance(val, list):
                new_list = [_poison_value(knob, ev) for ev in val]
                for j, nv in enumerate(new_list):
                    pos = positions.get((knob, j))
                    if pos is not None:
                        flat[pos] = nv
                cfg[knob] = new_list
            else:
                nv = _poison_value(knob, val)
                pos = positions.get((knob, 0))
                if pos is not None:
                    flat[pos] = nv
                if nv != val:
                    cfg[knob] = nv
        poisoned["flat_config"] = json.dumps(flat)
    else:
        for knob, val in list(cfg.items()):
            if isinstance(val, list):
                cfg[knob] = [_poison_value(knob, ev) for ev in val]
            else:
                cfg[knob] = _poison_value(knob, val)

    poisoned["config"] = json.dumps(cfg, indent=2)
    entry_path.write_text(json.dumps(poisoned, indent=2))
    return poisoned
