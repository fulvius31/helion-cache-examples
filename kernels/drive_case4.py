"""In-process Case-4 steps 1-3: tune -> poison -> degraded exact hit.

Single process => the kernel's default lambda keeps a stable cache key (the
exact-match key embeds the lambda's code-object address, which differs across
processes), so the poisoned probe is a real exact-match HIT against the entry
this process wrote.

Emits RESULT_JSON with the truth + poisoned + hit data for case4.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import best_perf, poison_entry  # noqa: E402


def snap(counters) -> dict:
    a = counters.get("autotune", {})
    return {
        "hit": a.get("cache_hit", 0),
        "miss": a.get("cache_miss", 0),
        "put": a.get("cache_put", 0),
    }


def main() -> None:
    logging.getLogger("helion").setLevel(logging.DEBUG)
    from helion._utils import counters
    from helion._testing import DEVICE, HALF_DTYPE
    from kernels.matmul_demo import matmul

    cache_dir = Path(os.environ["HELION_CACHE_DIR"])
    size = 4096  # compute-bound: poisoned block sizes measurably degrade perf
    x = torch.randn([size, size], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([size, size], device=DEVICE, dtype=HALF_DTYPE)

    out = {}  # structured results

    # --- step 1: tune (truth) ---
    b0 = snap(counters)
    torch.testing.assert_close(matmul(x, y), torch.matmul(x, y), rtol=1e-2, atol=1e-1)
    truth_cnt = snap(counters)
    truth_perf = best_perf(matmul, x, y, iters=30, warmup=5)
    entries = sorted(p.name for p in cache_dir.glob("*.best_config"))
    assert len(entries) == 1, f"expected 1 entry, got {entries}"
    entry_path = cache_dir / entries[0]
    truth_entry = json.loads(entry_path.read_text())
    truth_config = json.loads(truth_entry["config"])
    out["truth"] = {
        "perf_ms": truth_perf,
        "config": truth_config,
        "entry": entries[0],
        "delta": {k: truth_cnt[k] - b0[k] for k in ("hit", "miss", "put")},
    }
    print(f"  [1] tune (truth): perf={truth_perf:.4f} ms  config block_sizes={truth_config.get('block_sizes')} num_warps={truth_config.get('num_warps')}", flush=True)

    # --- step 2: poison the entry in place ---
    # (before-hash is captured AFTER poisoning so "unchanged" compares the
    #  poisoned file before vs after the exact-hit probe)
    poisoned = poison_entry(entry_path)
    entry_before_hash = hashlib.sha256(entry_path.read_bytes()).hexdigest()
    poisoned_config = poisoned["config"]
    poisoned_cfg = json.loads(poisoned_config)
    out["poisoned"] = {"config": poisoned_cfg}
    print(f"  [2] poisoned: block_sizes {truth_config.get('block_sizes')} -> {poisoned_cfg.get('block_sizes')}, num_warps {truth_config.get('num_warps')} -> {poisoned_cfg.get('num_warps')}", flush=True)

    # --- step 3: reset + exact-hit probe (same process -> stable key) -------
    matmul.reset()
    b1 = snap(counters)
    torch.testing.assert_close(matmul(x, y), torch.matmul(x, y), rtol=1e-2, atol=1e-1)
    hit_cnt = snap(counters)
    hit_perf = best_perf(matmul, x, y, iters=30, warmup=5)
    entry_after_hash = hashlib.sha256(entry_path.read_bytes()).hexdigest()
    out["hit"] = {
        "perf_ms": hit_perf,
        "delta": {k: hit_cnt[k] - b1[k] for k in ("hit", "miss", "put")},
        "entry_unchanged": entry_before_hash == entry_after_hash,
    }
    print(f"  [3] poisoned probe: perf={hit_perf:.4f} ms  counters_delta={out['hit']['delta']}  entry_unchanged={out['hit']['entry_unchanged']}", flush=True)

    print("RESULT_JSON=" + json.dumps(out))
    print("RUN_OK")


if __name__ == "__main__":
    main()
