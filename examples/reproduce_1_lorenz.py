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
    _run("runners.run_lorenz")
    print("\nDone: Lorenz benchmark (Table 1-style output printed to console).")


if __name__ == "__main__":
    main()