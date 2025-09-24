from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyvisgen.simulation import Visibilities

__all__ = ["compute_single_stokes_component", "get_stokes_from_vis_data"]


def get_stokes_from_vis_data(
    vis_data: Visibilities, stokes_comp: str, polarization: str
):
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
):
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
