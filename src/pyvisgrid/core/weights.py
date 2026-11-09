import numpy as np

from pyvisgrid.core.stokes import get_stokes_from_vis_data


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


def get_stokes_weights_from_vis_data(
    vis_data: np.ndarray, stokes_comp: str, polarization: str
) -> np.ndarray:
    """Noise weights of a Stokes visibility computed with
    :func:`~pyvisgrid.core.stokes.get_stokes_from_vis_data`.

    The Stokes visibility is a linear combination ``S = sum_k a_k V_k`` of the
    correlations ``V_k``. With independent noise its variance is
    ``1 / w_S = sum_k |a_k|^2 / w_k``. The coefficients ``a_k`` are obtained by
    evaluating ``get_stokes_from_vis_data`` on unit vectors, so the weights
    always follow the same Stokes convention (indices, signs, factors)
    as the visibilities.

    Parameters
    ----------
    vis_data : numpy.ndarray
        Visibility data with the correlations on the second to last axis and
        ``(real, imag, weight)`` on the last axis.

    stokes_comp : str
        The Stokes component, e.g. ``'I'``.

    polarization : str
        The polarization type, e.g. ``'linear'`` or ``'circular'``.

    Returns
    -------
    numpy.ndarray
        The Stokes weights, squeezed like the output of
        ``get_stokes_from_vis_data``. Samples where a contributing correlation
        has a weight <= 0 (flagged) get weight 0.
    """
    corr_weights = np.clip(
        np.asarray(vis_data[..., 2], dtype=np.float64), a_min=0.0, a_max=None
    )
    n_corr = corr_weights.shape[-1]

    # coefficient a_k of correlation k: Stokes visibility of the unit vector e_k
    basis = np.zeros((n_corr, n_corr, 3))
    basis[np.arange(n_corr), np.arange(n_corr), 0] = 1.0
    coeffs = np.abs(get_stokes_from_vis_data(basis, stokes_comp, polarization)) ** 2
    used = coeffs > 0

    # variance per correlation, infinite for flagged (zero weight) samples
    inv_w = np.divide(
        1.0,
        corr_weights[..., used],
        out=np.full(corr_weights[..., used].shape, np.inf),
        where=corr_weights[..., used] > 0,
    )
    variance = (inv_w * coeffs[used]).sum(axis=-1)

    return np.squeeze(1.0 / variance)
