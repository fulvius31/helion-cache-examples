"""Case 5 — remote cache backend: exact-match sharing, warm-start list(), graceful degradation.

All sub-cases run IN ONE PROCESS (the exact-match key embeds the default
lambda's code-object address, which differs across processes; one process
keeps it stable); machine A/B/B' differ by cache dir + cache class (switched
via the kernel's Settings), and `kernel.reset()` forces each "machine" to
consult the on-disk cache. Evidence is counted across the captured stderr log.

  5a machine A writes through to FileBackend; B reads through + materializes
     locally; B' (default LocalAutotuneCache) hits locally.
  5b edited kernel, exact-miss everywhere, warm-start seeded from list().
  5c FailingBackend full run completes; corrupt remote entry is skipped.
"""

from __future__ import annotations

from common import (
    fresh_cache_dir,
    python_bin,
    run_once,
)


def count(text: str, needle: str) -> int:
    return text.count(needle)


def main() -> int:
    py = python_bin()
    print("=" * 78)
    print("Case 5 — remote cache backend (in-process machines A/B/B' + 5b + 5c)")
    print("=" * 78)

    # fresh per-machine local cache dirs
    for name in ("case5_A", "case5_B", "case5_C", "case5_D", "case5_C_corrupt"):
        fresh_cache_dir(name)

    r = run_once(
        [py, "kernels/drive_case5.py"],
        env={"HELION_CACHE_DIR": str(fresh_cache_dir("case5_A")),
             "HELION_AUTOTUNE_EFFORT": "quick"},
        log="case5_all",
        timeout=3600,
    )
    assert r.ok, f"case5 driver failed rc={r.returncode}\n{r.output[-4000:]}"
    d = r.result_json() or {}
    A, B, Bp, C, D, Cc = (d.get(k, {}) for k in ("A", "B", "Bp", "C", "D", "Ccorrupt"))
    out = r.output

    print("\n[5a exact-match sharing]")
    print(f"  A (tune+write-through): {A}")
    print(f"  B (remote read+materialize): {B}")
    print(f"  B' (local no-backend): {Bp}")

    print("\n[5b warm-start list(), edited kernel]")
    print(f"  C (edited, default cache class): {C}")

    print("\n[5c graceful degradation]")
    print(f"  D (full w/ FailingBackend): {D}")
    print(f"  bonus corrupt store: {Cc}")

    # ---- evidence (counted over the combined log) ---------------------------
    using_cached = count(out, "Using cached config:")
    from_prev = count(out, "Found 1 cached config(s) from previous runs")
    transferred = count(out, "Cached config 1 (transferred)")
    read_fail = count(out, "Remote cache read failed")
    write_fail = count(out, "Remote cache write failed")
    print("\n[evidence]")
    ok5a = (A.get("remote_files", 0) >= 1 and B.get("local_entries", 0) >= 1
            and using_cached >= 2 and B.get("counters", {}).get("hit") == 1
            and Bp.get("counters", {}).get("hit") == 1)
    ok5b = (C.get("local_entries", 0) >= 1 and from_prev >= 2 and transferred >= 2
            and C.get("counters", {}).get("miss") == 1)
    ok5c = D.get("local_entries", 0) >= 1 and read_fail >= 1 and write_fail >= 1
    okbonus = Cc.get("local_entries", 0) >= 1 and from_prev >= 2
    print(f"  [{'OK' if ok5a else 'FAIL'}] 5a: A write-through ({A.get('remote_files')} remote file(s)), B local materialization + 2 exact hits (B, B')")
    print(f"  [{'OK' if ok5b else 'FAIL'}] 5b: edited kernel exact-miss + warm-start from FileBackend.list() (found/transferred: {from_prev}/{transferred})")
    print(f"  [{'OK' if ok5c else 'FAIL'}] 5c: full run completed, local entry written; read-fail ×{read_fail}, write-fail ×{write_fail}")
    print(f"  [{'OK' if okbonus else 'FAIL'}] bonus: corrupt remote JSON skipped, valid entry still seeded")

    passed = ok5a and ok5b and ok5c and okbonus
    print("\n[case 5 verdict]")
    print(f"  exact hits seen (B+B'): {using_cached};  warm-start seeds (C+bonus): {from_prev}")
    print(f"  graceful-degradation warnings: read {read_fail}, write {write_fail}")
    print(f"  local entries: A={A.get('local_entries')} B={B.get('local_entries')} C={C.get('local_entries')} D={D.get('local_entries')}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
