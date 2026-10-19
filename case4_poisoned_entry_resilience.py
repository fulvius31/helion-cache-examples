"""Case 4 — disruptive cache, untainted search (poisoned entry + forced re-tune).

Steps 1-3 (tune -> poison -> degraded exact hit) run IN ONE PROCESS so the
poisoned probe is a real exact hit (the exact-match key embeds the default
lambda's code-object address, which differs across processes; one process
keeps it stable).
Step 4 (forced full warm re-tune) is a fresh subprocess; warm-start is stable
cross-process so the poisoned seed is read, benchmarked, loses, and a good
winner is written. A new exact-match key can leave the old entry in place.

PASS: step 3 shows real degradation on the poisoned hit + the entry is
unchanged (hit path never writes); step 4 returns the truth (or better) and
writes a good config.
"""

from __future__ import annotations

import json

from common import (
    dump_cache_entries,
    fresh_cache_dir,
    python_bin,
    run_once,
)

NOISE = 0.05


def main() -> int:
    py = python_bin()
    print("=" * 78)
    print("Case 4 — disruptive cache, untainted search (poisoned entry + forced re-tune)")
    print("=" * 78)

    cache = fresh_cache_dir("case4")
    env_q = {"HELION_CACHE_DIR": str(cache), "HELION_AUTOTUNE_EFFORT": "quick"}

    # --- steps 1-3: tune, poison, degraded exact hit -------------------------
    r13 = run_once(
        [py, "kernels/drive_case4.py"], env=env_q, log="case4_1_3_hitpath", timeout=1800
    )
    assert r13.ok, f"case4 steps 1-3 failed rc={r13.returncode}\n{r13.output[-3000:]}"
    d = r13.result_json() or {}
    truth = d.get("truth", {})
    hit = d.get("hit", {})
    poisoned = d.get("poisoned", {})

    print("\n[1-3: tune -> poison -> degraded hit]")
    tcfg = truth.get("config", {})
    pcfg = poisoned.get("config", {})
    print(f"  truth   : perf={truth.get('perf_ms'):.4f} ms  blocks={tcfg.get('block_sizes')} warps={tcfg.get('num_warps')}")
    print(f"  poisoned: blocks={pcfg.get('block_sizes')} warps={pcfg.get('num_warps')}")
    print(f"  hit     : perf={hit.get('perf_ms'):.4f} ms  counters_delta={hit.get('delta')}  entry_unchanged={hit.get('entry_unchanged')}")
    print(f"  exact-hit evidence: {'Using cached config:' in r13.output}")

    degraded = hit.get("perf_ms", 0) > truth.get("perf_ms", 0) * (1 + NOISE)
    unchanged = bool(hit.get("entry_unchanged"))
    before = dump_cache_entries(cache)

    # --- step 4: forced full warm re-tune (repair) ---------------------------
    env_repair = {
        "HELION_CACHE_DIR": str(cache),
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_FORCE_AUTOTUNE": "1",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available",
        # Uncapped, matching the recorded 4096-cubed comparison.
    }
    print("\n[4] forced full warm re-tune @4096 (no budget cap)")
    r4 = run_once(
        [py, "kernels/run_matmul_demo.py", "--module", "matmul_demo",
         "--m", "4096", "--k", "4096", "--n", "4096"],
        env=env_repair, log="case4_4_repair_full", timeout=7200)
    assert r4.ok, f"repair run failed rc={r4.returncode}\n{r4.output[-4000:]}"
    repaired_perf = r4.perfs()
    after = dump_cache_entries(cache)
    written = [name for name, data in after.items() if before.get(name) != data]
    assert len(written) == 1, f"expected one new/updated entry, got {written}"
    repaired_name = written[0]
    repaired_cfg = json.loads(after[repaired_name]["config"])
    print(f"  best {repaired_perf:.4f} ms, wall {r4.elapsed_s:.1f}s, entry {repaired_name}")
    print(f"  repaired config blocks={repaired_cfg.get('block_sizes')} warps={repaired_cfg.get('num_warps')}")

    seeded = "Found 1 cached config(s) from previous runs" in r4.stderr
    logged_transfer = "Cached config 1 (transferred)" in r4.stderr
    logged_dedup = "Cached config 1 is a duplicate, skipping" in r4.stderr
    no_exact = "Using cached config:" not in r4.output
    repaired_matches_truth = repaired_cfg == tcfg
    perf_restored = repaired_perf <= truth.get("perf_ms", 0) * (1 + NOISE)
    entry_written = bool(written)
    cfg_changed = repaired_cfg != pcfg   # winner != poisoned config

    print("\n[evidence]")
    print(f"  [{'OK' if degraded else 'FAIL'}] poisoned exact-hit is measurably slower ({hit.get('perf_ms'):.4f} vs {truth.get('perf_ms'):.4f} ms)")
    print(f"  [{'OK' if unchanged else 'FAIL'}] entry file unchanged after the hit (hit path never writes)")
    print(f"  [{'OK' if seeded or logged_transfer or logged_dedup else 'FAIL'}] forced full warm seeds the (poisoned) cached config (found/transfer/dedup: {seeded}/{logged_transfer}/{logged_dedup})")
    print(f"  [{'OK' if no_exact else 'FAIL'}] forced re-tune does NOT exact-hit (it re-searches)")
    print(f"  [{'OK' if perf_restored or repaired_matches_truth else 'FAIL'}] repair returns the truth (or better)")
    print(f"  [{'OK' if cfg_changed and entry_written else 'FAIL'}] good config written to the cache")

    seed_ok = seeded or logged_transfer or logged_dedup
    passed = degraded and unchanged and seed_ok and no_exact and (perf_restored or repaired_matches_truth) and cfg_changed and entry_written
    print("\n[case 4 verdict]")
    print(f"  truth {truth.get('perf_ms'):.4f} -> poisoned hit {hit.get('perf_ms'):.4f} -> repaired {repaired_perf:.4f} ms")
    print(f"  repaired == truth config: {repaired_matches_truth}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
