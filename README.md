# Lyapunov Exponent — Robust Numerical Estimation for Chaotic Bubble Dynamics

**Paper:** *Robust numerical calculation of Lyapunov exponents for chaotic bubble dynamics*  
**DOI:** (to be added upon publication)

---

## Overview

This repository provides a robust numerical framework for computing Lyapunov exponents in nonlinear dynamical systems, with particular emphasis on acoustically driven bubble dynamics.

The implementation combines:

- Analytical Jacobians for:
  - Rayleigh–Plesset
  - Keller–Miksis
  - Gilmore
- Benettin QR re-orthonormalization
- Stabilized classical alternatives (Eckmann, Rosenstein)
- Validation against the Lorenz system

The repository serves two purposes:

1. A reusable numerical library for Lyapunov exponent estimation.
2. A fully reproducible environment for all tables and figures in the paper.

---

## Repository Structure

```
core/          Lyapunov engine, QR propagation, Jacobians
models/        Lorenz and bubble dynamical systems
experiments/   Parameter sweeps and comparison logic
plotting/      Phase portraits, Lyapunov maps, D2 maps
scripts/       Internal utilities and grid generation
configs/       Parameter configurations
examples/      Paper reproduction scripts (Tables & Figures)
results/       Generated numerical outputs (created automatically)
figures/       Generated figures (created automatically)
```

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/Lyapunov-Exponent.git
cd Lyapunov-Exponent
```

### 2) Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (recommended for clean imports):

```bash
pip install -e .
```

---

## Reproducing the Paper Results

All published tables and figures can be reproduced via scripts in the `examples/` directory.

---

### Tables

**Table 1 — Lorenz benchmark**
```bash
python examples/reproduce_table_1_lorenz.py
```

**Table 3 — Numerical comparison**
```bash
python examples/reproduce_table_3_numerical_comparison.py
```

**Table 4 — Reliability table**
```bash
python examples/reproduce_table_4_reliability.py
```

**Table 6 — Method comparison**
```bash
python examples/reproduce_table_6_method_comparison.py
```

**Table 7 — Configuration C3**
```bash
python examples/reproduce_table_7_config_c3.py
```

---

### Figures

**Figures 1–2 — Phase portraits**
```bash
python examples/reproduce_fig_1_2_phase_portraits.py
```

**Figure 3 — Lyapunov maps**

If grid files do not yet exist, generate them first:

```bash
python scripts/run_fixed_frequency_scan.py
python scripts/run_fixed_pressure_scan.py
```

Then generate the figure:

```bash
python examples/reproduce_fig_3_lyapunov_maps.py
```

---

## Output Files

Generated outputs are saved under:

```
results/
figures/
```

Grid files (`*.npy`) are reused automatically if already present.

---

## Using as a Library

After installing with:

```bash
pip install -e .
```

You may import core components directly:

```python
from core.lyapunov import compute_lyapunov_exponents
from models.lorenz import lorenz_system
```

This allows the framework to be used independently of the paper scripts.

---

## Citation

If you use this code in academic work, please cite:

> Robust numerical calculation of Lyapunov exponents for chaotic bubble dynamics  
> DOI: (to be added)

A BibTeX entry will be added once available.


---

## Maintainer

Trinidad Gatica  
Western University  
Department of Statistical and Actuarial Sciences