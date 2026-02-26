#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run C3 (Gilmore) and estimate Lyapunov exponents from the first 1061 points
using a robust grid search for Eckmann (spectrum) and Rosenstein (LLE).

This script is self-contained: it defines the full grid-search + helpers here,
and only relies on your existing simulation + LCE implementations:
- models.bubble_models.create_trajectories
- models.lorenz.compute_lce_eckmann
- models.lorenz.compute_lce_rosenstein
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Iterable, List, Tuple

from models.bubble_models import create_trajectories
from models.lorenz import (
    compute_lce_eckmann,       # must accept (x, dt, params) and return spectrum per second
    compute_lce_rosenstein     # must accept (x, dt, params) and return lle per second
)

# ===================== GRID / SEARCH KNOBS =====================
MATRIX_DIM = 2                                   # Eckmann: ask for two exponents
EMB_GRID: Tuple[int, ...] = (5, 6, 7, 8)         # Takens for bubbles (d≈2 ⇒ m≥5)
TAU_GRID_BASE: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40)
THEILER_MULTS: Tuple[int, ...] = (1, 2, 3)       # min_tsep = k * tau (will be capped)
TRAJ_LEN_FRACS: Tuple[float, ...] = (0.15, 0.25, 0.35)
USE_LOG_RADIUS = False                            # set True if collapse spikes dominate


# ===================== UTILITIES =====================
def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / (sd + 1e-12)


def _window_slices(n: int, win: int, step: int):
    i = 0
    while i + win <= n:
        yield slice(i, i + win)
        i += step


def _stability_cv(vals: np.ndarray) -> float:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.inf
    mu, sd = float(np.mean(v)), float(np.std(v))
    return float(sd / (abs(mu) + 1e-12))


def _make_tau_grid(dt: float, drive_freq_hz: float,
                   base: Iterable[int]) -> List[int]:
    """Base grid plus points near the period (T/6..T/12)."""
    per = max(1, int(round((1.0 / float(drive_freq_hz)) / dt)))
    extra = {max(1, int(round(per / d))) for d in (12, 10, 8, 6)}
    return sorted(set(base) | extra)


# ===================== CORE WRAPPERS (using your fns) =====================
def _eckmann_spec(x: np.ndarray, dt: float, emb_dim: int, tau: int, min_tsep: int) -> np.ndarray:
    params = dict(
        emb_dim=emb_dim,
        matrix_dim=MATRIX_DIM,
        tau=tau,
        min_tsep=min_tsep,
        min_nb=max(MATRIX_DIM + 1, 3),
    )
    return compute_lce_eckmann(x, dt, params)


def _rosenstein_lle(x: np.ndarray, dt: float, emb_dim: int, tau: int, min_tsep: int, trajectory_len: int) -> float:
    params = dict(
        emb_dim=emb_dim,
        tau=tau,
        min_tsep=min_tsep,
        trajectory_len=trajectory_len,
        fit="RANSAC",
    )
    return compute_lce_rosenstein(x, dt, params)


# ===================== GRID SEARCH (robust) =====================
def find_best_params_grid(
    radius: np.ndarray,
    dt: float,
    drive_freq_hz: float,
    emb_grid: Iterable[int] = EMB_GRID,
    tau_grid_base: Iterable[int] = TAU_GRID_BASE,
    theiler_mults: Iterable[int] = THEILER_MULTS,
    traj_len_fracs: Iterable[float] = TRAJ_LEN_FRACS,
    use_log_radius: bool = USE_LOG_RADIUS
) -> Dict[str, Any]:

    R = np.asarray(radius, float)
    x = np.log(np.clip(R, np.finfo(float).tiny, None)) if use_log_radius else (R - np.mean(R))
    x = _standardize(x)
    n = len(x)
    if n < 500:
        raise ValueError("Series too short (<500 samples). Provide at least 500.")

    # Geometry from period and dataset length
    period_samples = max(1, int(round((1.0 / drive_freq_hz) / dt)))
    win = min(max(2 * period_samples, 600), max(200, n // 3))   # ~2 periods, ≥600, ≤ n//3
    step = max(100, win // 2)
    wins = list(_window_slices(n, win, step))
    if len(wins) < 2:
        mid = n // 2
        wins = [slice(0, mid), slice(max(0, mid - win // 2), n)]

    # Tau candidates: base + near period
    tau_candidates = _make_tau_grid(dt, drive_freq_hz, tau_grid_base)

    # Cap Rosenstein trajectory length by window and series
    traj_len_fracs = tuple(sorted(set(traj_len_fracs)))

    def _cap_traj_len(tl: int) -> int:
        return int(min(max(20, tl), win // 2, n // 3))

    # Feasibility checks tuned for short-dt, few-period runs
    def _feasible(emb_dim: int, tau: int, min_tsep: int, traj_len: int) -> bool:
        max_delay = (emb_dim - 1) * tau
        if max_delay >= win - 10:  # must fit window
            return False
        if max_delay >= n - 10:    # must fit series
            return False
        if traj_len >= (win - max_delay - 10):
            return False
        if traj_len >= n // 2:
            return False
        # don't exclude everyone
        if min_tsep >= max(win // 2, n // 3):
            return False
        return True

    best: Dict[str, Any] = {"score": np.inf}

    # -------- search grid with soft Theiler caps & capped trajectory_len --------
    for tau in tau_candidates:
        base_theilers = [max(10, k * tau) for k in theiler_mults] + [period_samples]
        # Soft-cap Theiler so neighbors remain
        theilers = sorted(set(min(t, max(period_samples, n // 6)) for t in base_theilers))

        for m in emb_grid:
            if MATRIX_DIM > m:
                continue

            tl_candidates = sorted(set(_cap_traj_len(int(fr * n)) for fr in traj_len_fracs))
            if not tl_candidates:
                continue
            tl_mid = tl_candidates[len(tl_candidates) // 2]

            for theiler in theilers:
                if not _feasible(m, tau, theiler, tl_mid):
                    continue

                lam1_vals, lle_vals = [], []
                for sl in wins:
                    sig = x[sl]
                    # Eckmann λ1
                    try:
                        lam1_vals.append(float(_eckmann_spec(sig, dt, m, tau, theiler)[0]))
                    except Exception:
                        lam1_vals.append(np.nan)
                    # Rosenstein LLE
                    try:
                        lle_vals.append(float(_rosenstein_lle(sig, dt, m, tau, theiler, tl_mid)))
                    except Exception:
                        lle_vals.append(np.nan)

                lam1_vals = np.asarray(lam1_vals, float)
                lle_vals = np.asarray(lle_vals, float)
                if np.isnan(lam1_vals).mean() > 0.5 or np.isnan(lle_vals).mean() > 0.5:
                    continue

                score = 0.5 * (_stability_cv(lam1_vals[np.isfinite(lam1_vals)]) +
                               _stability_cv(lle_vals[np.isfinite(lle_vals)]))

                if score < best["score"]:
                    # pick Rosenstein tl by full-series median-by-value
                    lle_full_opts: List[Tuple[float, int]] = []
                    for tl in tl_candidates:
                        if not _feasible(m, tau, theiler, tl):
                            continue
                        try:
                            v = float(_rosenstein_lle(x, dt, m, tau, theiler, tl))
                            lle_full_opts.append((v, int(tl)))
                        except Exception:
                            pass
                    if not lle_full_opts:
                        continue
                    vals = np.array([v for v, _ in lle_full_opts], float)
                    tl_chosen = int(lle_full_opts[int(np.argsort(vals)[len(vals)//2])][1])

                    try:
                        spec_full = _eckmann_spec(x, dt, m, tau, theiler)
                    except Exception:
                        continue

                    best = {
                        "score": float(score),
                        "params": {
                            "dt": float(dt),
                            "emb_dim": int(m),
                            "matrix_dim": int(MATRIX_DIM),
                            "tau": int(tau),
                            "min_tsep": int(theiler),
                            "trajectory_len": int(tl_chosen),
                            "use_log_radius": bool(use_log_radius),
                            "drive_freq_hz": float(drive_freq_hz),
                            "window_len": int(win),
                            "window_step": int(step)
                        },
                        "eckmann": {"spectrum": np.asarray(spec_full, float)},
                        "rosenstein": {"lle": float(np.median(vals))},
                        "diagnostics": {"lam1_windows": lam1_vals, "lle_windows": lle_vals}
                    }

    # -------- robust fallback if grid found nothing --------
    if "params" not in best:
        tau_def = max(1, max(1, int(round((1.0 / drive_freq_hz) / dt))) // 10)
        m_def = min(max(EMB_GRID), max(MATRIX_DIM, min(EMB_GRID)))
        theiler_def = min(max(int(round((1.0 / drive_freq_hz) / dt)), 3 * tau_def),
                          max(int(round((1.0 / drive_freq_hz) / dt)), len(radius) // 6))
        tl_def = int(min(max(20, int(0.2 * len(radius))), (len(radius) // 3)))

        spec_full = _eckmann_spec(x, dt, m_def, tau_def, theiler_def)
        lle_full = _rosenstein_lle(x, dt, m_def, tau_def, theiler_def, tl_def)

        best = {
            "score": np.nan,
            "params": {
                "dt": float(dt),
                "emb_dim": int(m_def),
                "matrix_dim": int(MATRIX_DIM),
                "tau": int(tau_def),
                "min_tsep": int(theiler_def),
                "trajectory_len": int(tl_def),
                "use_log_radius": bool(use_log_radius),
                "drive_freq_hz": float(drive_freq_hz),
                "window_len": int(min(max(2 * int(round((1.0 / drive_freq_hz) / dt)), 600), max(200, len(radius) // 3))),
                "window_step": int(max(100, min(max(2 * int(round((1.0 / drive_freq_hz) / dt)), 600), max(200, len(radius) // 3)) // 2))
            },
            "eckmann": {"spectrum": np.asarray(spec_full, float)},
            "rosenstein": {"lle": float(lle_full)},
            "diagnostics": None
        }

    return best


# ===================== RUNNER FOR C3 ONLY (TRIM TO 1061) =====================
def main() -> None:
    # ------- Fixed setup -------
    equation = "G"         # 'RP', 'KM', 'G'
    temperature = 20       # Celsius
    periods = 10
    step = 1e-3            # integrator step control used by your runner (step/f)

    # ------- C3 only -------
    configuration_name = "C3"
    acoustic_pressure = 2.0e6   # Pa
    frequency = 0.8e6           # Hz
    initial_radius = 0.08e-6    # m

    print(f"\n=== Running {configuration_name} (trimmed to first 1061 points) ===")

    # Build time array exactly like your previous runner
    integration_time = np.arange(0, periods / frequency, step / frequency)

    # Create trajectories
    trajectories, _model = create_trajectories(
        [equation],
        temperature,
        acoustic_pressure,
        frequency,
        initial_radius,
        integration_time,
        step
    )

    # ----- Trim to first 1061 samples to avoid post-explosion data -----
    N_KEEP = 1061
    n_total = len(integration_time)
    if n_total < N_KEEP:
        raise ValueError(f"Series has only {n_total} points (< {N_KEEP}). "
                         f"Increase integration span or lower step to ensure at least {N_KEEP} samples.")

    t_used = integration_time[:N_KEEP]
    radius_full = trajectories[f"Radius_{equation}"]
    radius_used = np.asarray(radius_full, float)[:N_KEEP]

    # Recompute dt from the trimmed time vector
    dt_series = float(np.median(np.diff(t_used)))

    try:
        best = find_best_params_grid(
            radius=radius_used,
            dt=dt_series,
            drive_freq_hz=frequency,
            emb_grid=EMB_GRID,
            tau_grid_base=TAU_GRID_BASE,
            theiler_mults=THEILER_MULTS,
            traj_len_fracs=TRAJ_LEN_FRACS,
            use_log_radius=USE_LOG_RADIUS
        )

        lam1, lam2 = best["eckmann"]["spectrum"][:2]
        lle = best["rosenstein"]["lle"]
        chosen = best["params"]

        print("Chosen params:", chosen)
        print("Eckmann λ1, λ2 [1/s]:", lam1, " ", lam2)
        print("Rosenstein LLE [1/s]:", lle)

    except Exception as e:
        print(f"[{configuration_name}] Parameter search failed: {e}")
        print(f"  N={len(radius_used)}, dt={dt_series}, f={frequency} Hz")


if __name__ == "__main__":
    main()
