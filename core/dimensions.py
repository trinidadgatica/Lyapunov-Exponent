import numpy as np

def compute_lce_dimension_metrics(exponents: np.ndarray) -> dict:
    """
    Compute Kaplan–Yorke and approximate correlation dimension D₂ from Lyapunov exponents.

    Args:
        exponents (np.ndarray): Array-like of Lyapunov exponents [lambda_1, lambda_2].

    Returns:
        dict: Dictionary with 'Kaplan–Yorke' and 'Correlation D₂ (approx)' as keys.
    """
    lambda_1, lambda_2 = exponents

    if not np.isfinite(lambda_1) or not np.isfinite(lambda_2):
        return {"Kaplan–Yorke": np.nan, "Correlation D₂ (approx)": np.nan}

    if lambda_1 > 0 and lambda_2 != 0:
        kaplan_yorke_dim = 1 + lambda_1 / abs(lambda_2)
        correlation_d2 = lambda_1 / abs(lambda_2)
    else:
        kaplan_yorke_dim = 1.0
        correlation_d2 = 0.0

    return {"Kaplan–Yorke": kaplan_yorke_dim, "Correlation D₂ (approx)": correlation_d2}
