from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _ensure_dirs() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("figures/generated").mkdir(parents=True, exist_ok=True)


def _run_path(path: str) -> None:
    cmd = [sys.executable, path]
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    _ensure_dirs()

    _run_path("examples/reproduce_1_lorenz.py")
    _run_path("examples/reproduce_2_method_comparison.py")
    _run_path("examples/reproduce_3_phase_portraits.py")
    _run_path("examples/reproduce_4_lyapunov_maps.py")
    _run_path("examples/reproduce_5_d2_maps.py")

    print("\nAll paper reproduction steps finished.")


if __name__ == "__main__":
    main()