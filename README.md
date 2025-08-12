# Lyapunov-Exponent

Repository for the numerical analysis and computation of Lyapunov exponents in nonlinear bubble dynamics models.  
This work supports the study of acoustic cavitation regimes by providing robust, model-based algorithms for estimating Lyapunov exponents from physics-based simulations.

## Overview

This repository contains implementations of algorithms for computing Lyapunov exponents in bubble dynamics, focusing on three key physical models:

- **Rayleigh–Plesset (RP)** equation  
- **Keller–Miksis (KM)** equation  
- **Gilmore (G)** equation  

We compare several approaches, including:
- **QR-based Benettin algorithm** (trajectory-based)  
- **Eigenvalue product method** (from cumulative Jacobians)  
- **Log-determinant method** (sum of exponents)  

The goal is to provide numerically stable and accurate tools to classify cavitation regimes — stable, transient, and strongly unstable — across a wide range of physical parameters.  
These tools are useful in biomedical ultrasound, sonochemistry, and nonlinear acoustics research.

## Features

- Full numerical solvers for RP, KM, and G equations.
- Exact Jacobian computations for each model.
- Multiple Lyapunov exponent estimation methods for comparison.
- Grid-based parameter sweeps with automated logging and cut-off detection.
- Heatmap and time-series visualizations.
- Ready-to-use scripts for reproducible experiments in `runners/`.

## Installation

A recent Python version and the following packages are required:

```bash
conda install numpy scipy matplotlib pandas plotly
````
## Citation

If you use this repository in your research, please cite the following:

### BibTeX

```bibtex

````

### APA 

