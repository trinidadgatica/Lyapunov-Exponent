from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple  # noqa: UP035

import numpy as np

from models.bubble_models import simulate_bubble_trajectories
from models.lorenz import compute_lce_eckmann, compute_lce_rosenstein
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)

# ===================== GRID / SEARCH KNOBS =====================
MATRIX_DIM = 2
EMB_GRID: tuple[int, ...] = (5, 6, 7, 8)
TAU_GRID_BASE: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40)
THEILER_MULTS: tuple[int, ...] = (1, 2, 3)
TRAJ_LEN_FRACS: tuple[float, ...] = (0.15, 0.25, 0.35)
USE_LOG_RADIUS = False


def _zscore_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / (sd + 1e-12)


def _iter_window_slices(n: int, win: int, step: int):
    i = 0
    while i + win <= n:
        yield slice(i, i + win)
        i += step


def _coefficient_of_variation(vals: np.ndarray) -> float:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.inf
    mu, sd = float(np.mean(v)), float(np.std(v))
    return float(sd / (abs(mu) + 1e-12))


def _build_tau_grid(
    dt: float,
    drive_freq_hz: float,
    base: Iterable[int],
) -> list[int]:
    """Base grid plus points near the period (T/6..T/12)."""
    per = max(1, int(round((1.0 / float(drive_freq_hz)) / dt)))
    extra = {max(1, int(round(per / d))) for d in (12, 10, 8, 6)}
    return sorted(set(base) | extra)


def _compute_eckmann_spectrum(
    x: np.ndarray,
    dt: float,
    emb_dim: int,
    tau: int,
    min_tsep: int,
) -> np.ndarray:
    params = dict(
        emb_dim=emb_dim,
        matrix_dim=MATRIX_DIM,
        tau=tau,
        min_tsep=min_tsep,
        min_nb=max(MATRIX_DIM + 1, 3),
    )
    return compute_lce_eckmann(x, dt, params)


def search_best_nolds_parameters(
    x: np.ndarray,
    dt: float,
    emb_dim: int,
    tau: int,
    min_tsep: int,
    trajectory_len: int,
) -> float:
    params = dict(
        emb_dim=emb_dim,
        tau=tau,
        min_tsep=min_tsep,
        trajectory_len=trajectory_len,
        fit="RANSAC",
    )
    return compute_lce_rosenstein(x, dt, params)


def find_best_params_grid(
    radius: np.ndarray,
    dt: float,
    drive_freq_hz: float,
    emb_grid: Iterable[int] = EMB_GRID,
    tau_grid_base: Iterable[int] = TAU_GRID_BASE,
    theiler_mults: Iterable[int] = THEILER_MULTS,
    traj_len_fracs: Iterable[float] = TRAJ_LEN_FRACS,
    use_log_radius: bool = USE_LOG_RADIUS,
) -> Dict[str, Any]:
    logger.info(
        "Starting NOLDS parameter search: len(radius)=%d, dt=%s, drive_freq_hz=%s",
        len(radius),
        dt,
        drive_freq_hz,
    )

    R = np.asarray(radius, float)
    x = np.log(np.clip(R, np.finfo(float).tiny, None)) if use_log_radius else (R - np.mean(R))
    x = _zscore_standardize(x)
    n = len(x)

    if n < 500:
        logger.error("Series too short for parameter search: n=%d", n)
        raise ValueError("Series too short (<500 samples). Provide at least 500.")

    period_samples = max(1, int(round((1.0 / drive_freq_hz) / dt)))
    win = min(max(2 * period_samples, 600), max(200, n // 3))
    step = max(100, win // 2)
    wins = list(_iter_window_slices(n, win, step))

    if len(wins) < 2:
        mid = n // 2
        wins = [slice(0, mid), slice(max(0, mid - win // 2), n)]
        logger.warning("Using fallback windows because fewer than two windows were available.")

    logger.info(
        "Window settings: period_samples=%d, window_len=%d, window_step=%d, n_windows=%d",
        period_samples,
        win,
        step,
        len(wins),
    )

    tau_candidates = _build_tau_grid(dt, drive_freq_hz, tau_grid_base)
    logger.info("Generated %d tau candidates.", len(tau_candidates))

    traj_len_fracs = tuple(sorted(set(traj_len_fracs)))

    def _cap_traj_len(tl: int) -> int:
        return int(min(max(20, tl), win // 2, n // 3))

    def _feasible(emb_dim: int, tau: int, min_tsep: int, traj_len: int) -> bool:
        max_delay = (emb_dim - 1) * tau
        if max_delay >= win - 10:
            return False
        if max_delay >= n - 10:
            return False
        if traj_len >= (win - max_delay - 10):
            return False
        if traj_len >= n // 2:
            return False
        if min_tsep >= max(win // 2, n // 3):
            return False
        return True

    best: Dict[str, Any] = {"score": np.inf}
    combos_checked = 0
    feasible_combos = 0

    for tau_idx, tau in enumerate(tau_candidates, start=1):
        logger.info("Evaluating tau candidate %d/%d: tau=%d", tau_idx, len(tau_candidates), tau)

        base_theilers = [max(10, k * tau) for k in theiler_mults] + [period_samples]
        theilers = sorted(set(min(t, max(period_samples, n // 6)) for t in base_theilers))

        for m in emb_grid:
            if MATRIX_DIM > m:
                logger.warning("Skipping m=%d because MATRIX_DIM=%d > m.", m, MATRIX_DIM)
                continue

            tl_candidates = sorted(set(_cap_traj_len(int(fr * n)) for fr in traj_len_fracs))
            if not tl_candidates:
                logger.warning("No trajectory length candidates available for m=%d, tau=%d.", m, tau)
                continue

            tl_mid = tl_candidates[len(tl_candidates) // 2]

            for theiler in theilers:
                combos_checked += 1

                if not _feasible(m, tau, theiler, tl_mid):
                    continue

                feasible_combos += 1
                lam1_vals, lle_vals = [], []

                for win_idx, sl in enumerate(wins, start=1):
                    sig = x[sl]

                    try:
                        lam1_vals.append(float(_compute_eckmann_spectrum(sig, dt, m, tau, theiler)[0]))
                    except Exception as exc:
                        logger.warning(
                            "Eckmann failed for m=%d, tau=%d, theiler=%d, window=%d: %s",
                            m,
                            tau,
                            theiler,
                            win_idx,
                            exc,
                        )
                        lam1_vals.append(np.nan)

                    try:
                        lle_vals.append(float(search_best_nolds_parameters(sig, dt, m, tau, theiler, tl_mid)))
                    except Exception as exc:
                        logger.warning(
                            "Rosenstein failed for m=%d, tau=%d, theiler=%d, tl=%d, window=%d: %s",
                            m,
                            tau,
                            theiler,
                            tl_mid,
                            win_idx,
                            exc,
                        )
                        lle_vals.append(np.nan)

                lam1_vals = np.asarray(lam1_vals, float)
                lle_vals = np.asarray(lle_vals, float)

                if np.isnan(lam1_vals).mean() > 0.5 or np.isnan(lle_vals).mean() > 0.5:
                    continue

                score = 0.5 * (
                    _coefficient_of_variation(lam1_vals[np.isfinite(lam1_vals)])
                    + _coefficient_of_variation(lle_vals[np.isfinite(lle_vals)])
                )

                if score < best["score"]:
                    logger.info(
                        "New best score found: score=%s, m=%d, tau=%d, theiler=%d",
                        score,
                        m,
                        tau,
                        theiler,
                    )

                    lle_full_opts: List[Tuple[float, int]] = []
                    for tl in tl_candidates:
                        if not _feasible(m, tau, theiler, tl):
                            continue
                        try:
                            v = float(search_best_nolds_parameters(x, dt, m, tau, theiler, tl))
                            lle_full_opts.append((v, int(tl)))
                        except Exception as exc:
                            logger.warning(
                                "Full-series Rosenstein failed for m=%d, tau=%d, theiler=%d, tl=%d: %s",
                                m,
                                tau,
                                theiler,
                                tl,
                                exc,
                            )

                    if not lle_full_opts:
                        continue

                    vals = np.array([v for v, _ in lle_full_opts], float)
                    tl_chosen = int(lle_full_opts[int(np.argsort(vals)[len(vals) // 2])][1])

                    try:
                        spec_full = _compute_eckmann_spectrum(x, dt, m, tau, theiler)
                    except Exception as exc:
                        logger.warning(
                            "Full-series Eckmann failed for m=%d, tau=%d, theiler=%d: %s",
                            m,
                            tau,
                            theiler,
                            exc,
                        )
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
                            "window_step": int(step),
                        },
                        "eckmann": {"spectrum": np.asarray(spec_full, float)},
                        "rosenstein": {"lle": float(np.median(vals))},
                        "diagnostics": {
                            "lam1_windows": lam1_vals,
                            "lle_windows": lle_vals,
                        },
                    }

    logger.info(
        "Finished grid evaluation: combos_checked=%d, feasible_combos=%d",
        combos_checked,
        feasible_combos,
    )

    if "params" not in best:
        logger.warning("Grid search found no valid parameter set. Using robust fallback.")
        tau_def = max(1, period_samples // 10)
        m_def = min(max(EMB_GRID), max(MATRIX_DIM, min(EMB_GRID)))
        theiler_def = min(max(period_samples, 3 * tau_def), max(period_samples, n // 6))
        tl_def = _cap_traj_len(int(0.2 * n))

        spec_full = _compute_eckmann_spectrum(x, dt, m_def, tau_def, theiler_def)
        lle_full = search_best_nolds_parameters(x, dt, m_def, tau_def, theiler_def, tl_def)

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
                "window_len": int(win),
                "window_step": int(step),
            },
            "eckmann": {"spectrum": np.asarray(spec_full, float)},
            "rosenstein": {"lle": float(lle_full)},
            "diagnostics": None,
        }

    logger.info("Parameter search finished successfully.")
    return best


if __name__ == "__main__":
    equation = "G"
    temperature = 20
    periods = 10
    step = 1e-3

    configs = [
        ("C1", 0.3e6, 1.2e6, 10e-6),
        ("C2", 1.5e6, 1.2e6, 5e-6),
    ]

    logger.info("Starting standalone parameter search runner for %d configurations.", len(configs))

    for idx, (configuration_name, acoustic_pressure, frequency, initial_radius) in enumerate(configs, start=1):
        logger.info(
            "[%d/%d] Running %s with equation=%s, pressure=%s, frequency=%s, initial_radius=%s",
            idx,
            len(configs),
            configuration_name,
            equation,
            acoustic_pressure,
            frequency,
            initial_radius,
        )

        integration_time = np.arange(0, periods / frequency, step / frequency)
        dt_series = float(np.median(np.diff(integration_time)))

        try:
            trajectories, model = simulate_bubble_trajectories(
                [equation],
                temperature,
                acoustic_pressure,
                frequency,
                initial_radius,
                integration_time,
                step,
            )
            logger.info("%s: trajectories created successfully.", configuration_name)

            radius_data = trajectories[f"Radius_{equation}"]

            best = find_best_params_grid(
                radius=radius_data,
                dt=dt_series,
                drive_freq_hz=frequency,
                emb_grid=EMB_GRID,
                tau_grid_base=TAU_GRID_BASE,
                theiler_mults=THEILER_MULTS,
                traj_len_fracs=TRAJ_LEN_FRACS,
                use_log_radius=USE_LOG_RADIUS,
            )

            lam1, lam2 = best["eckmann"]["spectrum"][:2]
            lle = best["rosenstein"]["lle"]
            chosen = best["params"]

            logger.info("%s: chosen params = %s", configuration_name, chosen)
            logger.info("%s: Eckmann λ1, λ2 [1/s] = %s, %s", configuration_name, lam1, lam2)
            logger.info("%s: Rosenstein LLE [1/s] = %s", configuration_name, lle)

        except Exception:
            logger.exception(
                "%s: parameter search failed. N=%d, dt=%s, f=%s Hz",
                configuration_name,
                len(radius_data) if "radius_data" in locals() else -1,
                dt_series,
                frequency,
            )
            raise

    logger.info("Finished standalone parameter search runner.")