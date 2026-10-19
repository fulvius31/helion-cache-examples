"""Driver that runs a copied matmul kernel once in a fresh process.

Every case shells out to this script, so each step's environment (cache dir,
effort, force flags, remote backend...) is applied to a clean process and the
Helion autotune/cache log lines land on stderr where ``run_once`` captures
them.

Structured results are printed on stdout as ``RESULT_JSON=...``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import best_perf  # noqa: E402


def main() -> None:
    # Belt-and-braces on top of HELION_LOGS=+all: make sure the autotuner's
    # DEBUG population lines are visible even if logging config drifted.
    logging.getLogger("helion").setLevel(logging.DEBUG)

    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="matmul_demo")
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--reference", choices=("identity", "edited"), default="identity")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = importlib.import_module(f"kernels.{args.module}")
    kernel = mod.matmul
    dev, dtype = mod.DEVICE, mod.HALF_DTYPE

    x = torch.randn([args.m, args.k], device=dev, dtype=dtype)
    y = torch.randn([args.k, args.n], device=dev, dtype=dtype)

    # First call resolves the autotuned config (cache hit or fresh search).
    out = kernel(x, y)

    if args.reference == "edited":
        ref = torch.matmul(x, y) * 2.0
    else:
        ref = torch.matmul(x, y)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)

    ms = best_perf(kernel, x, y, iters=args.iters, warmup=args.warmup)
    result: dict[str, object] = {
        "best_perf_ms": ms,
        "shape": [args.m, args.k, args.n],
    }
    try:
        from helion._utils import counters

        result["counters"] = dict(counters.get("autotune", {}))
    except Exception:  # pragma: no cover - counters always present
        pass
    print("RESULT_JSON=" + json.dumps(result))
    print("RUN_OK")


if __name__ == "__main__":
    main()
