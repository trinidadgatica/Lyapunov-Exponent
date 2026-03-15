from __future__ import annotations

import typing
from collections.abc import Iterable

import numpy as np

from models.lorenz import compute_eckmann_lce, compute_rosenstein_lle
from utils.logging_utils import get_logger

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
    mu = float(np.mean(x))
    sd = float(np.std(x))
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
    mu = float(np.mean(v))
    sd = float(np.std(v))
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
    return compute_eckmann_lce(x, dt, params)


def _compute_rosenstein_lle(
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
    return compute_rosenstein_lle(x, dt, params)


def find_best_params_grid(
    radius: np.ndarray,
    dt: float,
    drive_freq_hz: float,
    emb_grid: Iterable[int] = EMB_GRID,
    tau_grid_base: Iterable[int] = TAU_GRID_BASE,
    theiler_mults: Iterable[int] = THEILER_MULTS,
    traj_len_fracs: Iterable[float] = TRAJ_LEN_FRACS,
    use_log_radius: bool = USE_LOG_RADIUS,
) -> typing.Dict[str, typing.Any]:
    logger.info(
        "Starting NOLDS parameter search: len(radius)=%d, dt=%s, drive_freq_hz=%s",
        len(radius),
        dt,
        drive_freq_hz,
    )

    radius_array = np.asarray(radius, float)
    if use_log_radius:
        signal = np.log(np.clip(radius_array, np.finfo(float).tiny, None))
    else:
        signal = radius_array - np.mean(radius_array)

    signal = _zscore_standardize(signal)
    num_samples = len(signal)

    if num_samples < 500:
        logger.error("Series too short for parameter search: n=%d", num_samples)
        raise ValueError("Series too short (<500 samples). Provide at least 500.")

    period_samples = max(1, int(round((1.0 / drive_freq_hz) / dt)))
    window_len = min(max(2 * period_samples, 600), max(200, num_samples // 3))
    window_step = max(100, window_len // 2)
    windows = list(_iter_window_slices(num_samples, window_len, window_step))

    if len(windows) < 2:
        mid = num_samples // 2
        windows = [
            slice(0, mid),
            slice(max(0, mid - window_len // 2), num_samples),
        ]
        logger.warning(
            "Using fallback windows because fewer than two windows were available."
        )

    logger.info(
        "Window settings: period_samples=%d, window_len=%d, window_step=%d, n_windows=%d",
        period_samples,
        window_len,
        window_step,
        len(windows),
    )

    tau_candidates = _build_tau_grid(dt, drive_freq_hz, tau_grid_base)
    logger.info("Generated %d tau candidates.", len(tau_candidates))

    traj_len_fracs = tuple(sorted(set(traj_len_fracs)))

    def _cap_traj_len(traj_len: int) -> int:
        return int(min(max(20, traj_len), window_len // 2, num_samples // 3))

    def _feasible(
        emb_dim: int,
        tau: int,
        min_tsep: int,
        trajectory_len: int,
    ) -> bool:
        max_delay = (emb_dim - 1) * tau
        if max_delay >= window_len - 10:
            return False
        if max_delay >= num_samples - 10:
            return False
        if trajectory_len >= (window_len - max_delay - 10):
            return False
        if trajectory_len >= num_samples // 2:
            return False
        if min_tsep >= max(window_len // 2, num_samples // 3):
            return False
        return True

    best: typing.Dict[str, typing.Any] = {"score": np.inf}
    combos_checked = 0
    feasible_combos = 0

    for tau_idx, tau in enumerate(tau_candidates, start=1):
        logger.info(
            "Evaluating tau candidate %d/%d: tau=%d",
            tau_idx,
            len(tau_candidates),
            tau,
        )

        base_theilers = [max(10, k * tau) for k in theiler_mults] + [period_samples]
        theilers = sorted(set(min(t, max(period_samples, num_samples // 6)) for t in base_theilers))

        for emb_dim in emb_grid:
            if MATRIX_DIM > emb_dim:
                logger.warning(
                    "Skipping emb_dim=%d because MATRIX_DIM=%d > emb_dim.",
                    emb_dim,
                    MATRIX_DIM,
                )
                continue

            traj_len_candidates = sorted(
                set(_cap_traj_len(int(fr * num_samples)) for fr in traj_len_fracs)
            )
            if not traj_len_candidates:
                logger.warning(
                    "No trajectory length candidates available for emb_dim=%d, tau=%d.",
                    emb_dim,
                    tau,
                )
                continue

            traj_len_mid = traj_len_candidates[len(traj_len_candidates) // 2]

            for theiler in theilers:
                combos_checked += 1

                if not _feasible(emb_dim, tau, theiler, traj_len_mid):
                    continue

                feasible_combos += 1
                lam1_vals = []
                lle_vals = []

                for win_idx, window_slice in enumerate(windows, start=1):
                    window_signal = signal[window_slice]

                    try:
                        lam1_vals.append(
                            float(
                                _compute_eckmann_spectrum(
                                    window_signal,
                                    dt,
                                    emb_dim,
                                    tau,
                                    theiler,
                                )[0]
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Eckmann failed for emb_dim=%d, tau=%d, theiler=%d, window=%d: %s",
                            emb_dim,
                            tau,
                            theiler,
                            win_idx,
                            exc,
                        )
                        lam1_vals.append(np.nan)

                    try:
                        lle_vals.append(
                            float(
                                _compute_rosenstein_lle(
                                    window_signal,
                                    dt,
                                    emb_dim,
                                    tau,
                                    theiler,
                                    traj_len_mid,
                                )
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Rosenstein failed for emb_dim=%d, tau=%d, theiler=%d, trajectory_len=%d, window=%d: %s",
                            emb_dim,
                            tau,
                            theiler,
                            traj_len_mid,
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
                        "New best score found: score=%s, emb_dim=%d, tau=%d, theiler=%d",
                        score,
                        emb_dim,
                        tau,
                        theiler,
                    )

                    lle_full_options: typing.List[typing.Tuple[float, int]] = []
                    for traj_len in traj_len_candidates:
                        if not _feasible(emb_dim, tau, theiler, traj_len):
                            continue
                        try:
                            lle_value = float(
                                _compute_rosenstein_lle(
                                    signal,
                                    dt,
                                    emb_dim,
                                    tau,
                                    theiler,
                                    traj_len,
                                )
                            )
                            lle_full_options.append((lle_value, int(traj_len)))
                        except Exception as exc:
                            logger.warning(
                                "Full-series Rosenstein failed for emb_dim=%d, tau=%d, theiler=%d, trajectory_len=%d: %s",
                                emb_dim,
                                tau,
                                theiler,
                                traj_len,
                                exc,
                            )

                    if not lle_full_options:
                        continue

                    lle_option_values = np.array(
                        [value for value, _ in lle_full_options],
                        float,
                    )
                    chosen_idx = int(np.argsort(lle_option_values)[len(lle_option_values) // 2])
                    trajectory_len_chosen = int(lle_full_options[chosen_idx][1])

                    try:
                        spectrum_full = _compute_eckmann_spectrum(
                            signal,
                            dt,
                            emb_dim,
                            tau,
                            theiler,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Full-series Eckmann failed for emb_dim=%d, tau=%d, theiler=%d: %s",
                            emb_dim,
                            tau,
                            theiler,
                            exc,
                        )
                        continue

                    best = {
                        "score": float(score),
                        "params": {
                            "dt": float(dt),
                            "emb_dim": int(emb_dim),
                            "matrix_dim": int(MATRIX_DIM),
                            "tau": int(tau),
                            "min_tsep": int(theiler),
                            "trajectory_len": int(trajectory_len_chosen),
                            "use_log_radius": bool(use_log_radius),
                            "drive_freq_hz": float(drive_freq_hz),
                            "window_len": int(window_len),
                            "window_step": int(window_step),
                        },
                        "eckmann": {"spectrum": np.asarray(spectrum_full, float)},
                        "rosenstein": {"lle": float(np.median(lle_option_values))},
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
        tau_default = max(1, period_samples // 10)
        emb_dim_default = min(max(EMB_GRID), max(MATRIX_DIM, min(EMB_GRID)))
        theiler_default = min(
            max(period_samples, 3 * tau_default),
            max(period_samples, num_samples // 6),
        )
        trajectory_len_default = _cap_traj_len(int(0.2 * num_samples))

        spectrum_full = _compute_eckmann_spectrum(
            signal,
            dt,
            emb_dim_default,
            tau_default,
            theiler_default,
        )
        lle_full = _compute_rosenstein_lle(
            signal,
            dt,
            emb_dim_default,
            tau_default,
            theiler_default,
            trajectory_len_default,
        )

        best = {
            "score": np.nan,
            "params": {
                "dt": float(dt),
                "emb_dim": int(emb_dim_default),
                "matrix_dim": int(MATRIX_DIM),
                "tau": int(tau_default),
                "min_tsep": int(theiler_default),
                "trajectory_len": int(trajectory_len_default),
                "use_log_radius": bool(use_log_radius),
                "drive_freq_hz": float(drive_freq_hz),
                "window_len": int(window_len),
                "window_step": int(window_step),
            },
            "eckmann": {"spectrum": np.asarray(spectrum_full, float)},
            "rosenstein": {"lle": float(lle_full)},
            "diagnostics": None,
        }

    logger.info("Parameter search finished successfully.")
    return best