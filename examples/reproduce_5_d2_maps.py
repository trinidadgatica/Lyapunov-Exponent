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

    # Compute grids first (needed for D2 map generation scripts as well)
    _run("runners.run_fixed_frequency_scan")
    _run("runners.run_fixed_pressure_scan")

    # Then generate D2 maps
    _run("runners.run_correlation_dimension_maps")

    print("\nDone: correlation dimension (D2) maps.")


if __name__ == "__main__":
    main()