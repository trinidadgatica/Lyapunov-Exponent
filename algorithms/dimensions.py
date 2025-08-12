import numpy as np

def compute_lyapunov_dimensions(exponents):
    """
    Compute Kaplan–Yorke and approximate correlation dimension D₂ from Lyapunov exponents.
    """
    λ1, λ2 = exponents

    if not np.isfinite(λ1) or not np.isfinite(λ2):
        return {"Kaplan–Yorke": np.nan, "Correlation D₂ (approx)": np.nan}

    if λ1 > 0 and λ2 != 0:
        d_ky = 1 + λ1 / abs(λ2)
        d2 = λ1 / abs(λ2)
    else:
        d_ky = 1.0
        d2 = 0.0

    return {"Kaplan–Yorke": d_ky, "Correlation D₂ (approx)": d2}
