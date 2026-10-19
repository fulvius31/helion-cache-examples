"""Case 1 — Source-edit iteration: exact-match dies, warm-start survives.

Steps:
  1. quick-tune the *unedited* kernel in a fresh cache dir (creates the entry)
  2. edit ``kernels/matmul_demo.py`` *inside the kernel body* (store line gets
     a ``* 2.0``), which changes ``kernel_source_hash`` without touching the
     tiling skeleton or config space
  3. same cache dir, quick effort, edited kernel -> exact match MISS, but the
     winner from step 1 is warm-started from the cache
  4. control: fresh cache dir, edited kernel, quick -> cold (seeds-only)
  5. restore the kernel file

PASS: warm perf >= cold control perf (within noise) and log evidence of the
cached seed being transferred into the warm population.
"""

from __future__ import annotations

from common import (
    apply_matmul_edit,
    check_evidence,
    dump_cache_entries,
    entry_keys,
    fresh_cache_dir,
    python_bin,
    restore_matmul,
    run_once,
)

NOISE = 0.03  # generous 3% noise band for B200 timing drift


def main() -> int:
    py = python_bin()
    runner = ["kernels/run_matmul_demo.py", "--module", "matmul_demo"]

    print("=" * 78)
    print("Case 1 — source-edit iteration: exact-match dies, warm-start survives")
    print("=" * 78)

    try:
        # --- step 1: quick-tune the identity kernel (fresh cache) ----------
        cache1 = fresh_cache_dir("case1")
        env1 = {
            "HELION_CACHE_DIR": str(cache1),
            "HELION_AUTOTUNE_EFFORT": "quick",
        }
        r1 = run_once([py, *runner], env=env1, log="case1_1_quick_tune")
        assert r1.ok, f"step 1 failed rc={r1.returncode}\n{r1.output[-3000:]}"
        print("\n[1a] quick-tune identity kernel (fresh cache)")
        print(f"  wall: {r1.elapsed_s:.1f}s  best: {r1.perfs():.4f} ms")
        print(f"  counters: {r1.counters()}")
        print(f"  cache entries: {entry_keys(cache1)}")
        step1_perf = r1.perfs()
        before = dump_cache_entries(cache1)
        assert len(before) == 1, "expected one baseline cache entry"
        original_fields = next(iter(before.values()))["key"]["fields"]

        # --- step 2: edit the kernel body in place -------------------------
        apply_matmul_edit()
        print("\n[1b] edited kernels/matmul_demo.py store line -> *2.0")

        # --- step 3: same cache dir, edited kernel, quick (warm) -----------
        r3 = run_once(
            [py, *runner, "--reference", "edited"],
            env={"HELION_CACHE_DIR": str(cache1), "HELION_AUTOTUNE_EFFORT": "quick"},
            log="case1_3_edited_warm",
        )
        assert r3.ok, f"step 3 failed rc={r3.returncode}\n{r3.output[-3000:]}"
        print("\n[1c] edited kernel, SAME cache dir, quick (warm-start run)")
        print(f"  wall: {r3.elapsed_s:.1f}s  best: {r3.perfs():.4f} ms")
        print(f"  counters: {r3.counters()}")
        print(f"  cache entries: {entry_keys(cache1)}")
        warm_perf = r3.perfs()
        after = dump_cache_entries(cache1)
        new_entries = [entry for name, entry in after.items() if name not in before]
        assert len(new_entries) == 1, "expected one entry for the edited kernel"
        edited_fields = new_entries[0]["key"]["fields"]
        source_changed = original_fields["kernel_source_hash"] != edited_fields["kernel_source_hash"]
        config_space_same = original_fields["config_spec_hash"] == edited_fields["config_spec_hash"]
        print(f"  source hash changed: {source_changed}; config-space hash unchanged: {config_space_same}")

        # --- step 4: fresh cache dir, edited kernel, quick (cold control) --
        cache4 = fresh_cache_dir("case1_control")
        r4 = run_once(
            [py, *runner, "--reference", "edited"],
            env={
                "HELION_CACHE_DIR": str(cache4),
                "HELION_AUTOTUNE_EFFORT": "quick",
            },
            log="case1_4_edited_cold",
        )
        assert r4.ok, f"step 4 failed rc={r4.returncode}\n{r4.output[-3000:]}"
        print("\n[1d] edited kernel, FRESH cache dir, quick (cold control)")
        print(f"  wall: {r4.elapsed_s:.1f}s  best: {r4.perfs():.4f} ms")
        print(f"  counters: {r4.counters()}")
        print(f"  cache entries: {entry_keys(cache4)}")
        cold_perf = r4.perfs()

        # --- evidence checks ----------------------------------------------
        print("\n[evidence]")
        no_exact = (
            "Using cached config:" not in r3.output
            and "skipping autotuning" not in r3.output
        )
        print(f"  [{'OK' if no_exact else 'FAIL'}] no exact hit on the edited kernel (1c)")
        found_debug = "Found 1 cached config(s) from previous runs" in r3.stderr
        transferred_debug = "Cached config 1 (transferred)" in r3.stderr
        print(f"  [{'OK' if found_debug else 'FAIL'}] warm-start found the cached seed: \"Found 1 cached config(s)..\" (1c)")
        print(f"  [{'OK' if transferred_debug else 'FAIL'}] cached config transferred into population: \"Cached config 1 (transferred)\" (1c)")
        no_cold_seeds = "from previous runs" not in r4.output
        print(f"  [{'OK' if no_cold_seeds else 'FAIL'}] cold control has no cached seeds (1d)")

        warm_pop = None
        cold_pop = None
        for line in r3.stderr.splitlines():
            if "Seed/default/cache population:" in line:
                warm_pop = line.strip()
        for line in r4.stderr.splitlines():
            if "Seed/default/cache population:" in line:
                cold_pop = line.strip()
        if warm_pop and cold_pop:
            print(f"  population sizes -> warm: {warm_pop}  cold: {cold_pop}")

        passed = all(
            [
                no_exact,
                source_changed,
                config_space_same,
                found_debug,
                transferred_debug,
                no_cold_seeds,
                warm_perf is not None
                and cold_perf is not None
                and warm_perf <= cold_perf * (1 + NOISE),
            ]
        )

        print("\n[case 1 verdict]")
        print(f"  step-1 (identity) best:      {step1_perf:.4f} ms")
        print(f"  warm edited best:            {warm_perf:.4f} ms")
        print(f"  cold edited best:            {cold_perf:.4f} ms")
        print(f"  warm >= cold within {NOISE:.0%}: {warm_perf <= cold_perf * (1 + NOISE)}")
        print(f"  exact MISS on edited kernel:  {no_exact}")
        print(f"  warm-start seed transferred:   {transferred_debug}")
        print(f"  cold control unseeded:         {no_cold_seeds}")
        print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        restore_matmul()
        print("\n[teardown] restored kernels/matmul_demo.py to the identity body")


if __name__ == "__main__":
    raise SystemExit(main())
