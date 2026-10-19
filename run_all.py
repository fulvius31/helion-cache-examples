"""Run the curated demos, save their console output, and summarize results.

Usage:  python run_all.py
"""

from __future__ import annotations

import subprocess

from common import ROOT, python_bin

CASES = [
    "case1_source_edit_warmstart.py",
    "case5_remote_cache_backend.py",
    "compare_full_then_quick.py",
    "case2_effort_escalation.py",
    "case4_poisoned_entry_resilience.py",
]


def main() -> int:
    py = python_bin()
    output_dir = ROOT / "results" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, int]] = []
    for case in CASES:
        print(f"\n\n########## RUNNING {case} ##########\n", flush=True)
        with (output_dir / case.replace(".py", ".out")).open("w") as output:
            proc = subprocess.Popen(
                [py, "-u", case], cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                output.write(line)
            results.append((case, proc.wait()))

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for case, rc in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {case} (rc={rc})")
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
