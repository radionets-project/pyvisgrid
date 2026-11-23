from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pyvisgen.simulation import Visibilities

__all__ = [
    "compute_single_stokes_component",
    "get_ms_data_idx",
    "get_stokes_from_vis_data",
]

STOKES_PARAMETER_FORMULAS = {
    "I": {
        "linear": "XX+YY",
        "circular": "RR+LL",
    },
    "V": {
        "linear": "i(YX-XY)",
        "circular": "RR-LL",
    },
    "Q": {
        "linear": "XX-YY",
        "circular": "RL-LR",
    },
    "U": {
        "linear": "XY+YX",
        "circular": "i(LR-RL)",
    },
}

# From http://github.com/casacore/casacore/blob/master/measures/Measures/Stokes.h
CORR_TYPES = [
    "Undefined",
    # standard stokes parameters
    "I",
    "Q",
    "U",
    "V",
    # circular correlation products
    "RR",
    "RL",
    "LR",
    "LL",
    # linear correlation products
    "XX",
    "XY",
    "YX",
    "YY",
    # mixed correlation products
    "RX",
    "RY",
    "LX",
    "LY",
    "XR",
    "XL",
    "YR",
    "YL",
    # general quasi-orthogonal correlation products
    "PP",
    "PQ",
    "QP",
    "QQ",
]

CORR_TYPES = dict(zip(CORR_TYPES, np.arange(len(CORR_TYPES)).tolist()))


def get_stokes_from_vis_data(
    vis_data: Visibilities, stokes_comp: str, polarization: str
) -> np.ndarray:
    """Get the Stokes visibility for a given Stokes component
    depending on polarization.

    Parameters
    ----------
    vis_data : pyvisgen.simulation.Visibilities
        The Visibilities which are the output of the ``pyvisgen.simulation.vis_loop``
        function.

    stokes_comp : str
        The name of the Stokes component which should be returned.
        Valid names are: ``'I'``, ``'Q'``, ``'U'``, ``'V'``, ``'I+V'``,
        ``'Q+U'``, ``'Q-U'``, ``'I-V'``.

    polarization : str
        The type of polarization which should be considered.
        Valid values are: ``'circular'`` and any value. In case this is not set to
        ``'circular'``, the result will be the same for all other values.

    Returns
    -------
    numpy.ndarray:
        The Stokes visibility of the given Stokes component with the given polarization.
    """
    if polarization == "circular":
        match stokes_comp:
            case "I":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "+")
            case "Q":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "+")
            case "U":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "-")
            case "V":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "-")
            case "I+V":
                stokes_vis = vis_data[..., 0, 0] + 1j * vis_data[..., 0, 1]
            case "Q+U":
                stokes_vis = vis_data[..., 1, 0] + 1j * vis_data[..., 1, 1]
            case "Q-U":
                stokes_vis = vis_data[..., 2, 0] + 1j * vis_data[..., 2, 1]
            case "I-V":
                stokes_vis = vis_data[..., 3, 0] + 1j * vis_data[..., 3, 1]
    else:
        match stokes_comp:
            case "I":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "+")
            case "Q":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "-")
            case "U":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "+")
            case "V":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "-")
            case "I+Q":
                stokes_vis = vis_data[..., 0, 0] + 1j * vis_data[..., 0, 1]
            case "U+V":
                stokes_vis = vis_data[..., 1, 0] + 1j * vis_data[..., 1, 1]
            case "U-V":
                stokes_vis = vis_data[..., 2, 0] + 1j * vis_data[..., 2, 1]
            case "I-Q":
                stokes_vis = vis_data[..., 3, 0] + 1j * vis_data[..., 3, 1]

    return np.squeeze(stokes_vis)


def compute_single_stokes_component(
    vis_data: Visibilities,
    stokes_comp_1: int,
    stokes_comp_2: int,
    sign: str,
) -> np.ndarray:
    """Computes single stokes components I, Q, U, or V from visibility
    data for gridding.

    Parameters
    ----------
    vis_data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` dataclass object
        containing the visibilities measured by the array.

    stokes_comp_1 : int
        Index of first stokes visibility.

    stokes_comp_2 : int
        Index of second stokes visibility.

    sign : str
        Whether to add or subtract ``stokes_comp_1`` and ``stokes_comp_2``.
        Valid values are ``'+'`` or ``'-'``.

    Returns
    -------
    numpy.ndarray
        The visibilities for the specified Stokes component.

    """
    if sign not in "+-":
        raise ValueError("'sign' can only be '+' or '-'!")
    match sign:
        case "+":
            real = vis_data[..., stokes_comp_1, 0] + vis_data[..., stokes_comp_2, 0]
            imag = vis_data[..., stokes_comp_1, 1] + vis_data[..., stokes_comp_2, 1]
        case "-":
            real = vis_data[..., stokes_comp_1, 0] - vis_data[..., stokes_comp_2, 0]
            imag = vis_data[..., stokes_comp_1, 1] - vis_data[..., stokes_comp_2, 1]

    return real + 1j * imag


def get_ms_data_idx(stokes_comp: str, corr_types: ArrayLike) -> dict:

    @dataclass
    class Correlation:
        name: str

        def get_corr_idx(self) -> int:
            return CORR_TYPES[self.name]

        def is_in_types(self) -> bool:
            return self.get_corr_idx() in corr_types

        def get_data_idx(self) -> int:
            return int(np.argwhere(corr_types == self.get_corr_idx()).ravel()[0])

    def _decode_stokes_formula(formula: str):
        imaginary = formula.startswith("i")
        signum = -1 if "-" in formula else 1
        correlations = (
            formula.replace("i", "")
            .replace("(", "")
            .replace(")", "")
            .split("-" if signum == -1 else "+")
        )
        return {
            "imaginary": imaginary,
            "signum": signum,
            "correlations": [Correlation(name) for name in correlations],
        }

    output = {
        "imaginary": False,
        "signum": 1,
        "data_idx": [None, None],
    }

    for mode in ["circular", "linear"]:
        calc_params = _decode_stokes_formula(
            STOKES_PARAMETER_FORMULAS[stokes_comp][mode]
        )
        valid = [
            correlation.is_in_types() for correlation in calc_params["correlations"]
        ]

        if not np.all(valid):
            continue

        output["imaginary"] = calc_params["imaginary"]
        output["signum"] = calc_params["signum"]
        output["data_idx"] = [
            correlation.get_data_idx() for correlation in calc_params["correlations"]
        ]

    if output["data_idx"] == [None, None]:
        raise IndexError(
            "The Measurement Set does not contain valid correlations to "
            f"calculate Stokes {stokes_comp}."
        )

    return output
