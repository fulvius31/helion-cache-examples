"""Full search -> quick warm (reuses the full-generated cache) -> quick cold.

Comparison the user asked for, with NO budget cap and a BIGGER matmul so the
full search is genuinely long:
  1. FULL  : cold full search in a fresh cache dir, no
             HELION_AUTOTUNE_BUDGET_SECONDS -> runs its full schedule
             (initial population + up to 20 generations + final selection).
  2. QUICK-WARM: quick effort against the SAME cache dir the full run filled.
               quick effort = from_best_available, so the full winner is
               warm-started into the initial population instead of tuning from
               nothing.
  3. QUICK-COLD: quick effort against a FRESH cache dir (no prior knowledge).

Dropping the optional HELION_AUTOTUNE_BUDGET_SECONDS cap exposes how long an
unconstrained full search actually takes on a big matmul.

PASS criteria:
  * quick-warm best perf <= full best perf * (1 + noise)  (reuses the winner)
  * quick-warm wall < full wall                            (much cheaper)
  * quick-warm found the cached seed ("Found 1 cached config(s)..")
   * quick-cold has no cached seed (timing is reported, not guaranteed)
"""

from __future__ import annotations

from common import (
    check_evidence,
    entry_keys,
    fresh_cache_dir,
    python_bin,
    run_once,
)

NOISE = 0.05
DEFAULT_SIZE = 4096  # big matmul -> each config compile/benchmark is slow, so the uncapped full search takes minutes


def main() -> int:
    import argparse as _parser
    _ap = _parser.ArgumentParser()
    _ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                     help="square matmul side length (m=k=n)")
    _ap.add_argument("--timeout-full", type=int, default=7200,
                     help="subprocess timeout for the full search, seconds")
    _args = _ap.parse_args()
    size = _args.size
    py = python_bin()
    runner = [
        "kernels/run_matmul_demo.py", "--module", "matmul_demo",
        "--m", str(size), "--k", str(size), "--n", str(size),
    ]

    print("=" * 78)
    print(f"Comparison: FULL (cold)  ->  QUICK warm  ->  QUICK cold   [{size}x{size}x{size}]")
    print("no HELION_AUTOTUNE_BUDGET_SECONDS (uncapped full search)")
    print("=" * 78)

    # --- 1. full cold (no budget cap) ---------------------------------------
    cache_full = fresh_cache_dir("cmp_full")
    env_full = {
        "HELION_CACHE_DIR": str(cache_full),
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_random",
        # NOTE: no HELION_AUTOTUNE_BUDGET_SECONDS -> full runs to completion
    }
    print("\n[1] FULL cold search (fresh cache, from_random, NO budget cap)")
    r_full = run_once([py, *runner], env=env_full, log="cmp_1_full_cold", timeout=_args.timeout_full)
    assert r_full.ok, f"full failed rc={r_full.returncode}\n{r_full.output[-4000:]}"
    full_perf = r_full.perfs()
    print(f"    wall {r_full.elapsed_s:.1f}s  best {full_perf:.4f} ms  entries {entry_keys(cache_full)}")

    # --- 2. quick warm (uses the cache the full run generated) --------------
    env_q_warm = {
        "HELION_CACHE_DIR": str(cache_full),
        "HELION_AUTOTUNE_EFFORT": "quick",
        # Re-tune even if the exact-match key is stable across processes.
        "HELION_FORCE_AUTOTUNE": "1",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available",
    }
    print("\n[2] QUICK warm (SAME cache dir as full -> warm-starts the winner)")
    r_qw = run_once([py, *runner], env=env_q_warm, log="cmp_2_quick_warm", timeout=3600)
    assert r_qw.ok, f"quick warm failed rc={r_qw.returncode}\n{r_qw.output[-3000:]}"
    qw_perf = r_qw.perfs()
    print(f"    wall {r_qw.elapsed_s:.1f}s  best {qw_perf:.4f} ms  entries {entry_keys(cache_full)}")

    # --- 3. quick cold ------------------------------------------------------
    cache_qc = fresh_cache_dir("cmp_quick_cold")
    env_q_cold = {
        "HELION_CACHE_DIR": str(cache_qc),
        "HELION_AUTOTUNE_EFFORT": "quick",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available",
    }
    print("\n[3] QUICK cold (FRESH cache dir, no prior knowledge)")
    r_qc = run_once([py, *runner], env=env_q_cold, log="cmp_3_quick_cold", timeout=3600)
    assert r_qc.ok, f"quick cold failed rc={r_qc.returncode}\n{r_qc.output[-3000:]}"
    qc_perf = r_qc.perfs()
    print(f"    wall {r_qc.elapsed_s:.1f}s  best {qc_perf:.4f} ms  entries {entry_keys(cache_qc)}")

    # --- evidence -----------------------------------------------------------
    print("\n[evidence]")
    seed_seen = check_evidence(
        r_qw, "Found 1 cached config(s) from previous runs", "quick-warm found the full winner in cache"
    )
    transferred = "Cached config 1 (transferred)" in r_qw.stderr
    deduped = "Cached config 1 is a duplicate, skipping" in r_qw.stderr
    seed_outcome = "transferred" if transferred else ("duplicate-of-seed" if deduped else "absent")
    print(f"  [{'OK' if transferred or deduped else 'FAIL'}] cached winner {'transferred into population' if transferred else 'was a duplicate of a compiler seed (deduped, logged)'}")
    cold_not_seeded = "from previous runs" not in r_qc.output
    print(f"  [{'OK' if cold_not_seeded else 'FAIL'}] quick-cold is unseeded")

    def qline(r, pat):
        for line in r.stderr.splitlines():
            if pat in line:
                return line.strip()
        return None

    full_qual = qline(r_full, "Qualified population of ")
    qw_qual = qline(r_qw, "Qualified population of ")
    qc_qual = qline(r_qc, "Qualified population of ")
    if full_qual:
        print(f"  FULL       {full_qual}")
    if qw_qual:
        print(f"  QUICK-WARM {qw_qual}")
    if qc_qual and qc_qual != qw_qual:
        print(f"  QUICK-COLD {qc_qual}")

    quick_warm_as_good = (
        qw_perf is not None and full_perf is not None and qw_perf <= full_perf * (1 + NOISE)
    )
    quick_cold_no_better = (
        qc_perf is not None and qw_perf is not None and qc_perf >= qw_perf * (1 - NOISE)
    )
    much_cheaper = r_qw.elapsed_s < r_full.elapsed_s * 0.9

    print("\n[verdict]")
    print(f"  FULL      best {full_perf:.4f} ms   wall {r_full.elapsed_s:.1f}s")
    print(f"  QUICK-WARM best {qw_perf:.4f} ms   wall {r_qw.elapsed_s:.1f}s")
    print(f"  QUICK-COLD best {qc_perf:.4f} ms   wall {r_qc.elapsed_s:.1f}s")
    print(f"  quick-warm >= full within {NOISE:.0%}:     {quick_warm_as_good}")
    print(f"  quick-warm cheaper than full (wall): {much_cheaper}")
    print(f"  quick-cold not better than quick-warm: {quick_cold_no_better}")
    print(f"  quick-warm cache outcome: {seed_outcome}")

    passed = (
        quick_warm_as_good
        and much_cheaper
        and seed_seen
        and (transferred or deduped)
        and cold_not_seeded
        and "Using cached config:" not in r_qw.output
    )
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
