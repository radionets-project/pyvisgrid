import numpy as np


def read_ms_weights(tab, shape: tuple[int, int, int]) -> np.ndarray:
    """Noise weights of shape (nrow, nchan, ncorr):
    WEIGHT_SPECTRUM → WEIGHT (broadcast over channels) → 1/SIGMA²."""
    cols = tab.colnames()
    if "WEIGHT_SPECTRUM" in cols and tab.iscelldefined("WEIGHT_SPECTRUM", 0):
        return tab.getcol("WEIGHT_SPECTRUM").astype(np.float64)
    if "WEIGHT" in cols:
        w = tab.getcol("WEIGHT").astype(np.float64)
    else:
        sigma = tab.getcol("SIGMA").astype(np.float64)
        w = np.divide(1.0, sigma**2, out=np.zeros_like(sigma), where=sigma > 0)
    return np.broadcast_to(w[:, None, :], shape).copy()

def combine_weights(w_a: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """Inverse-variance weight of V_a ± V_b, i.e. 1/w = 1/w_a + 1/w_b.
    Zero if either input is zero (flagged)."""
    w_a = np.asarray(w_a, dtype=np.float64)
    w_b = np.asarray(w_b, dtype=np.float64)
    denom = w_a + w_b
    return np.divide(
        w_a * w_b, denom, out=np.zeros_like(denom), where=(w_a > 0) & (w_b > 0)
    )
