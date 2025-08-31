# Robust numerical calculation of Lyapunov exponents for chaotic bubble dynamics

> **Paper:** **Robust numerical calculation of Lyapunov exponents for chaotic bubble dynamics**  
> **DOI:** _coming soon_

---

## Abstract
Bubble dynamics are highly nonlinear and can exhibit complex behaviors, including transitions to chaos. A standard approach for characterizing the dynamics is to calculate the Lyapunov exponents, which are a measure of sensitivity to initial conditions and identify the final states of the bubbles. The numerical methods available for calculating Lyapunov exponents often lack numerical stability when applied to intricate dynamical regimes of bubble oscillations, involving large oscillations. To address this limitation, we have developed a new algorithm that combines the Benettin method with QR orthonormalization and incorporates analytical Jacobians derived from the main bubble dynamics models; including the Rayleigh–Plesset, Keller–Miksis, and Gilmore equations. Benchmark comparisons against established numerical methods and validation with the Lorenz system demonstrate that our approach achieves stable and consistent results across a range of conditions and complex bubble dynamics, highlighting its reliability for the study of chaotic behavior in bubble dynamics.

---

## Installation (recommended: isolated `.venv`).

### 1) Clone the repo
```bash
git clone https://github.com/<your-org>/Lyapunov-Exponent.git
cd Lyapunov-Exponent
```

### 2) Create & activate a virtual environment

**Windows (PowerShell)**
```powershell
# from repo root
python -m venv .venv

# then activate
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

### 4) (Optional but recommended) Editable install for clean imports
```bash
pip install -e .
```
After this, imports like `from algorithms.lyapunov import compute_lyapunov_exponents_from_trajectory` work from anywhere inside the repo.

---

## Configuration

### Results directory
By default, results are written to `results/`. You can override this via an environment variable:

**Windows (PowerShell):**
```powershell
$env:LYAP_RESULTS_DIR = "C:\\path\\to\\Lyapunov-Exponent\\results"
```

**macOS / Linux:**
```bash
export LYAP_RESULTS_DIR="/absolute/path/to/Lyapunov-Exponent/results"
```

(If unset, the library falls back to `./results`. Make sure the folder exists or is creatable.)

---

## Citing this work
If you use this library or its results, please cite the paper:

> **Robust numerical calculation of Lyapunov exponents for chaotic bubble dynamics**.  
> **DOI:** _coming soon_

A BibTeX entry will be added here once the DOI is assigned.

---

## Acknowledgments

