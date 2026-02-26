from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _ensure_dirs() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("figures/generated").mkdir(parents=True, exist_ok=True)


def _run(module: str) -> None:
    cmd = [sys.executable, "-m", module]
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    _ensure_dirs()

    # 1) generate the grid data required by the plotting script
    _run("scripts.run_fixed_frequency_scan")
    _run("scripts.run_fixed_pressure_scan")

    # 2) plot LE maps from saved *.npy grids
    _run("scripts.run_lyapunov_maps")

    print("\nDone: Lyapunov maps (computed grids + plotted figures).")


if __name__ == "__main__":
    main()