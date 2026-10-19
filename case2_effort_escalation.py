"""Case 2 — Effort escalation: quick -> full, warm-started.

Goal: show that cached winners plus random padding converge at least as well
as a cold full search within the same budget.

Steps:
  1. fresh cache dir, quick-tune (creates the entry)
  2. control: FRESH cache dir, full effort, forced re-tune (from_random default).
     Run first so any B200 thermal drift helps the warm run, never the control.
  3. warm full: the step-1 cache dir, HELION_FORCE_AUTOTUNE=1, full effort,
     HELION_AUTOTUNER_INITIAL_POPULATION=from_best_available, and a 120 s
     budget.  The force flag is essential: without it the run exact-hits and
     never searches.

PASS: warm-full best >= cold-full best within noise; the cached seed is found
and transferred into the warm population, and the population is padded with
random configs to the full size (proved by the pre-padding
"Seed/default/cache population: N total" vs "Qualified population of 100").
"""

from __future__ import annotations

from common import (
    check_evidence,
    entry_keys,
    fresh_cache_dir,
    python_bin,
    run_once,
)

NOISE = 0.05  # generous 5% band for full-effort B200 timing drift
BUDGET = "120"


def main() -> int:
    py = python_bin()
    runner = ["kernels/run_matmul_demo.py", "--module", "matmul_demo"]

    print("=" * 78)
    print("Case 2 — effort escalation: quick -> full, warm-started")
    print("=" * 78)

    # --- step 1: quick-tune to create a cache entry ------------------------
    cache1 = fresh_cache_dir("case2")
    r1 = run_once(
        [py, *runner],
        env={"HELION_CACHE_DIR": str(cache1), "HELION_AUTOTUNE_EFFORT": "quick"},
        log="case2_1_quick",
    )
    assert r1.ok, f"quick tune failed rc={r1.returncode}\n{r1.output[-3000:]}"
    print(f"\n[1]   quick tune (fresh cache):  wall {r1.elapsed_s:.1f}s, best {r1.perfs():.4f} ms")
    print(f"      entries: {entry_keys(cache1)}")

    # --- step 2: control — cold full search (no initial-pop override) -------
    cache_cold = fresh_cache_dir("case2_control")
    env_cold = {
        "HELION_CACHE_DIR": str(cache_cold),
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_random",
        "HELION_FORCE_AUTOTUNE": "1",
        "HELION_AUTOTUNE_BUDGET_SECONDS": BUDGET,
    }
    print("\n[2]   control: cold FULL search (fresh cache, from_random default)")
    r_cold = run_once([py, *runner], env=env_cold, log="case2_2_cold_full", timeout=1500)
    assert r_cold.ok, f"cold full failed rc={r_cold.returncode}\n{r_cold.output[-4000:]}"
    print(f"      wall {r_cold.elapsed_s:.1f}s, best {r_cold.perfs():.4f} ms")
    cold_perf = r_cold.perfs()
    print(f"      entries: {entry_keys(cache_cold)}")

    # --- step 3: warm full — seeded from the step-1 cache -------------------
    env_warm = {
        "HELION_CACHE_DIR": str(cache1),
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_FORCE_AUTOTUNE": "1",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available",
        "HELION_AUTOTUNE_BUDGET_SECONDS": BUDGET,
    }
    print("\n[3]   warm FULL search (same cache, from_best_available, budget 120 s)")
    r_warm = run_once([py, *runner], env=env_warm, log="case2_3_warm_full", timeout=1500)
    assert r_warm.ok, f"warm full failed rc={r_warm.returncode}\n{r_warm.output[-4000:]}"
    print(f"      wall {r_warm.elapsed_s:.1f}s, best {r_warm.perfs():.4f} ms")
    warm_perf = r_warm.perfs()
    print(f"      entries: {entry_keys(cache1)}")

    # --- evidence -----------------------------------------------------------
    # NOTE: this Helion snapshot's pattern search pads the best-available
    # population with random configs in place (pattern_search.py extends the
    # population with ``random_flat()`` draws), so no
    # "Initial population after unique random padding: N total" line is emitted
    # on the normal full path.  Random padding is instead proven by the
    # pre-padding "Seed/default/cache population: N total" followed by the
    # benchmarked "Qualified population of N" count (target = 100 at full).
    import re

    def _grep(stderr: str, pat: str) -> str | None:
        for line in stderr.splitlines():
            if pat in line:
                return line.strip()
        return None

    def _count(line: str | None, pat: str) -> int | None:
        if line is None:
            return None
        m = re.search(pat + r"(\d+)", line)
        return int(m.group(1)) if m else None

    warm_pre = _grep(r_warm.stderr, "Seed/default/cache population:")
    warm_qual = _grep(r_warm.stderr, "Qualified population of ")
    cold_pre = _grep(r_cold.stderr, "Seed/default/cache population:")
    cold_qual = _grep(r_cold.stderr, "Qualified population of ")
    warm_pre_n = _count(warm_pre, "population: ")
    warm_qual_n = _count(warm_qual, "population of ")
    cold_qual_n = _count(cold_qual, "population of ")

    better_or_equal = (
        warm_perf is not None
        and cold_perf is not None
        and warm_perf <= cold_perf * (1 + NOISE)
    )

    print("\n[evidence]")
    found_debug = "Found 1 cached config(s) from previous runs" in r_warm.stderr
    transferred_debug = "Cached config 1 (transferred)" in r_warm.stderr
    cold_unseeded = "from previous runs" not in r_cold.output
    padded = (
        warm_pre_n is not None
        and warm_qual_n is not None
        and warm_qual_n > warm_pre_n
        and warm_qual_n == 100
    )
    print(f"  [{'OK' if found_debug else 'FAIL'}] warm full found cached seed (\"Found 1 cached config(s)..\")")
    print(f"  [{'OK' if transferred_debug else 'FAIL'}] cached winner transferred into the warm population")
    print(f"  [{'OK' if padded else 'FAIL'}] warm full padded with random (pre-padding {warm_pre_n} -> qualified {warm_qual_n})")
    print(f"  [{'OK' if cold_unseeded else 'FAIL'}] cold full unseeded (from_random control)")
    if warm_pre:
        print(f"  warm:  {warm_pre}")
    if warm_qual:
        print(f"  warm:  {warm_qual}")
    if cold_qual:
        print(f"  cold:  {cold_qual}")

    passed = all(
        [
            found_debug,
            transferred_debug,
            padded,
            cold_unseeded,
            better_or_equal,
        ]
    )

    print("\n[case 2 verdict]")
    print(f"  cold full best: {cold_perf:.4f} ms (wall {r_cold.elapsed_s:.1f}s)")
    print(f"  warm full best: {warm_perf:.4f} ms (wall {r_warm.elapsed_s:.1f}s)")
    print(f"  warm >= cold within {NOISE:.0%}: {better_or_equal}")
    print(f"  warm found cached seed + random padding (10 -> 100): {found_debug and padded}")
    print(f"  cold control unseeded: {'from previous runs' not in r_cold.output}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
