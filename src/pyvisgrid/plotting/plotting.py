import warnings

import astropy.units as units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_ungridded_uv", "plot_dirty_image", "plot_mask"]


def _configure_axes(
    fig: matplotlib.figure.Figure | None,
    ax: matplotlib.axes.Axes | None,
    fig_args: dict = None,
):
    """Configures figure and axis depending if they were given
    as parameters.

    If neither figure nor axis are given, a new subplot will be created.
    If they are given the given ones will be returned.
    If only one of both is not given, this will cause an exception.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | None
        The figure object.
    ax : matplotlib.axes.Axes | None
        The axes object.
    fig_args : dict, optional
        Optional arguments to be supplied to the ``plt.subplots`` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if fig_args is None:
        fig_args = {}

    if None in (fig, ax) and not all(x is None for x in (fig, ax)):
        raise KeyError("The parameters ax and fig have to be both None or not None!")

    if ax is None:
        fig, ax = plt.subplots(layout="tight", **fig_args)

    return fig, ax


def _get_norm(norm: str):
    """Converts a string parameter to a matplotlib norm.

    Parameters
    ----------
    norm : str
        The name of the norm.
        Possible values are:

        - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                            values above the maximum will be mapped to the maximum and
                            values below the minimum will be mapped to the minimum, thus
                            avoiding the appearance of a colormaps 'over' and 'under'
                            colors (e.g. in case of negative values). Depending on the
                            use case this is desirable but in case that it is not, one
                            can set the norm to ``log_noclip`` or provide a custom norm.

        - ``log_noclip``:   Returns a logarithmic norm with clipping off.

        - ``centered``:     Returns a linear norm which centered around zero.

        - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                            square-root of the values.

        - other:            A value not declared above will be returned as is, meaning
                            that this could be any value which exists in matplotlib
                            itself.

    Returns
    -------
    matplotlib.colors.Normalize | str
        The norm or the str if no specific norm is defined for the string.
    """
    match norm:
        case "log":
            return matplotlib.colors.LogNorm(clip=True)
        case "log_noclip":
            return matplotlib.colors.LogNorm(clip=False)
        case "centered":
            return matplotlib.colors.CenteredNorm()
        case "sqrt":
            return matplotlib.colors.PowerNorm(0.5)
        case _:
            return norm


def _apply_crop(ax: matplotlib.axes.Axes, crop: tuple[list[float | None]]):
    """Applies a specific x and y limit ('crop') to the given axis.
    This will effectively crop the image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis which to apply the limits to.
    crop : tuple[list[float | None]]
        The crop of the image. This has to have the format
        ``([x_left, x_right], [y_left, y_right])``, where the left and right
        values for each axis are the upper and lower limits of the axes which
        should be shown.
        IMPORTANT: If one supplies the ``plt.imshow`` an ``extent`` parameter,
        this will be the scale in which one has to give the crop! If not, the crop
        has to be in pixels.
    """
    ax.set_xlim(crop[0][0], crop[0][1])
    ax.set_ylim(crop[1][0], crop[1][1])


def plot_ungridded_uv(
    gridder,
    mode: str = "wave",
    show_times: bool = True,
    use_relative_time: bool = True,
    time_cmap: str | matplotlib.colors.Colormap = "inferno",
    colorbar_shrink: float = 1.0,
    marker_size: float | None = None,
    aspect_args: dict | None = None,
    plot_args: dict = None,
    fig_args: dict = None,
    save_to: str | None = None,
    save_args: dict = None,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
    """Plots the ungridded (u,v) points as a scatter plot.

    Parameters
    ----------
    gridder : pyvisgrid.Gridder
        The gridder from which to take the (u,v) coordinates.
    mode : str, optional
        The mode specifying the scale of the (u,v) coordinates.
        This can be either ``wave``, meaning the coordinates are
        plotted in units of the reference wavelength, or ``meter``,
        meaning the (u,v) coordinates will be plotted in meter.
        Default is ``wave``.
    show_times : bool, optional
        Whether to show the timestamps of the measured visibilities
        as a colormap. Default is ``True``.
    use_relative_time : bool, optional
        Whether to show the times relative to the timestamp of the
        first measurement in hours.
        Default is ``True``.
    times_cmap: str | matplotlib.colors.Colormap, optional
        The colormap to be used for the time component of the plot.
        Default is ``'inferno'``.
    colorbar_shrink: float, optional
        The shrink parameter of the colorbar. This can be needed if the plot is
        included as a subplot to adjust the size of the colorbar.
        Default is ``1``, meaning original scale.
    marker_size : float | None, optional
        The size of the scatter markers in points**2.
        Default is ``None``, meaning the default value supplied by
        your matplotlib rcParams.
    plot_args : dict, optional
        The additional arguments passed to the scatter plot.
        Default is ``{"color":"royalblue"}``.
    fig_args : dict, optional
        The additional arguments passed to the figure.
        If a figure object is given in the ``fig`` parameter, this
        value will be discarded.
        Default is ``{}``.
    save_to : str | None, optional
        The name of the file to save the plot to.
        Default is ``None``, meaning the plot won't be saved.
    save_args : dict, optional
        The additional arguments passed to the ``fig.savefig`` call.
        Default is ``{"bbox_inches":"tight"}``.
    fig : matplotlib.figure.Figure | None, optional
        A custom figure object.
        If set to ``None``, the ``ax`` parameter also has to be ``None``!
        Default is ``None``.
    ax : matplotlib.axes.Axes | None, optional
        A custom axes object.
        If set to ``None``, the ``fig`` parameter also has to be ``None``!
        Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if plot_args is None:
        plot_args = dict(color="royalblue") if not show_times else dict()

    if fig_args is None:
        fig_args = {}

    if save_args is None:
        save_args = dict(bbox_inches="tight")

    if aspect_args is None:
        aspect_args = dict(aspect="equal", adjustable="box")

    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    match mode:
        case "wave":
            u, v = gridder.u_wave, gridder.v_wave
            unit = "$\\lambda$"
        case "meter":
            u, v = gridder.u_meter, gridder.v_meter
            unit = "m"
        case _:
            raise ValueError(
                "The given mode does not exist! Valid modes are: wave, meter."
            )

    times = np.tile(gridder.times.mjd, reps=2) if show_times else None
    time_unit = "MJD"

    if use_relative_time and show_times:
        times -= times[0] * 24
        time_unit = "h"

    scat = ax.scatter(
        x=np.append(-u, u),
        y=np.append(-v, v),
        c=times,
        s=marker_size,
        cmap=time_cmap if show_times else None,
        **plot_args,
    )

    if show_times:
        fig.colorbar(scat, ax=ax, shrink=colorbar_shrink, label="Time / " + time_unit)

    ax.set_aspect(**aspect_args)
    scat.set_rasterized(True)

    ax.set_xlabel(f"$u$ / {unit}")
    ax.set_ylabel(f"$v$ / {unit}")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return fig, ax


def plot_mask(
    grid_data,
    mode: str = "hist",
    crop: tuple[list[float | None]] = ([None, None], [None, None]),
    norm: str | matplotlib.colors.Normalize = None,
    colorbar_shrink: float = 1,
    cmap: str | matplotlib.colors.Colormap | None = None,
    plot_args: dict = None,
    fig_args: dict = None,
    save_to: str | None = None,
    save_args: dict = None,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
    """Plots the (u,v) mask (the binned visibilities) of the gridded
    interferometric image.

    Parameters
    ----------
    grid_data : pyvisgrid.GridData
        The gridded data from the ``pyvisgrid.Gridder.grid`` method.
        This always represents the gridded visibilities of one
        Stokes component.
    mode : str, optional
        The mode specifying which values of the mask should be plotted.
        Possible values are:

        - ``hist``:     Plots the number of (u,v) points which are sorted in
                        each pixel of the image in the (u,v) space.

        - ``abs``:      Plots the absolute value of the gridded visibilities,
                        meaning the magnitude of the complex numbers in Euler
                        representation.

        - ``phase``:    Plots the phase angle of the gridded visibilities,
                        meaning the angle in the exponent of the complex numbers in
                        Euler representation.

        - ``real``:     Plots the real part of the gridded visibilities.

        - ``imag``:     Plots the imaginary part of the gridded visibilities.

        Default is ``hist``.
    crop : tuple[list[float | None]], optional
        The crop of the image. This has to have the format
        ``([x_left, x_right], [y_left, y_right])``, where the left and right
        values for each axis are the upper and lower limits of the axes which
        should be shown.
        IMPORTANT: If one supplies the ``plt.imshow`` an ``extent`` parameter
        via the ``plot_args`` parameter, this will be the scale in which one
        has to give the crop! If not, the crop has to be in pixels.
    norm : str | matplotlib.colors.Normalize | None, optional
        The name of the norm or a matplotlib norm.
        Possible values are:

        - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                            values above the maximum will be mapped to the maximum and
                            values below the minimum will be mapped to the minimum, thus
                            avoiding the appearance of a colormaps 'over' and 'under'
                            colors (e.g. in case of negative values).
                            Depending on the use case this is desirable but in case that
                            it is not, one can set the norm to ``log_noclip`` or provide
                            a custom norm.

        - ``log_noclip``:   Returns a logarithmic norm with clipping off.

        - ``centered``:     Returns a linear norm which centered around zero.

        - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                            square-root of the values.

        - other:            A value not declared above will be returned as is, meaning
                            that this could be any value which exists in matplotlib
                            itself.

        Default is ``None``, meaning no norm will be applied.
    colorbar_shrink: float, optional
        The shrink parameter of the colorbar. This can be needed if the plot is
        included as a subplot to adjust the size of the colorbar.
        Default is ``1``, meaning original scale.
    cmap: str | matplotlib.colors.Colormap | None, optional
        The colormap to be used for the plot.
        Default is ``None``, meaning the colormap will be default to a value
        fitting for the chosen mode.
    plot_args : dict, optional
        The additional arguments passed to the scatter plot.
        Default is ``{"color":"royalblue"}``.
    fig_args : dict, optional
        The additional arguments passed to the figure.
        If a figure object is given in the ``fig`` parameter, this
        value will be discarded.
        Default is ``{}``.
    save_to : str | None, optional
        The name of the file to save the plot to.
        Default is ``None``, meaning the plot won't be saved.
    save_args : dict, optional
        The additional arguments passed to the ``fig.savefig`` call.
        Default is ``{"bbox_inches":"tight"}``.
    fig : matplotlib.figure.Figure | None, optional
        A custom figure object.
        If set to ``None``, the ``ax`` parameter also has to be ``None``!
        Default is ``None``.
    ax : matplotlib.axes.Axes | None, optional
        A custom axes object.
        If set to ``None``, the ``fig`` parameter also has to be ``None``!
        Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if plot_args is None:
        plot_args = {}

    if fig_args is None:
        fig_args = {}

    if save_args is None:
        save_args = dict(bbox_inches="tight")

    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    cmap_dict = {
        "hist": "inferno",
        "abs": "viridis",
        "phase": "RdBu",
        "real": "RdBu",
        "imag": "RdBu",
    }

    cmap = cmap_dict[mode] if cmap is None else cmap

    norm = _get_norm(norm) if isinstance(norm, str) else norm

    match mode:
        case "hist":
            im = ax.imshow(
                grid_data.mask,
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            fig.colorbar(
                im, ax=ax, shrink=colorbar_shrink, label="$(u,v)$ per frequel / 1/fq"
            )
        case "abs":
            mask_abs, _ = grid_data.get_mask_abs_phase()
            im = ax.imshow(
                mask_abs,
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Amplitude / a.u.")
        case "phase":
            _, mask_phase = grid_data.get_mask_abs_phase()
            im = ax.imshow(
                mask_phase,
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            cbar = fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Phase / rad")

            cbar.set_ticks(np.arange(-np.pi, 3 / 2 * np.pi, np.pi / 2))
            cbar.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        case "real":
            im = ax.imshow(
                grid_data.mask_real,
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Real Part / a.u.")
        case "imag":
            im = ax.imshow(
                grid_data.mask_imag,
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            fig.colorbar(
                im, ax=ax, shrink=colorbar_shrink, label="Imaginary Part / a.u."
            )
        case _:
            raise ValueError(
                f"The given mode does not exist!"
                f"Valid modes are: {', '.join(list(cmap_dict.keys()))}"
            )

    ax.set_xlabel("Frequels")
    ax.set_ylabel("Frequels")

    _apply_crop(ax=ax, crop=crop)

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return fig, ax


def plot_dirty_image(
    grid_data,
    mode: str = "real",
    ax_unit: str | units.Unit = "pixel",
    center_pos: tuple[float] | None = None,
    norm: str | matplotlib.colors.Normalize = None,
    colorbar_shrink: float = 1,
    cmap: str | matplotlib.colors.Colormap = "inferno",
    plot_args: dict = None,
    fig_args: dict = None,
    save_to: str | None = None,
    save_args: dict = None,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
    """Plots the (u,v) dirty image, meaning the 2d Fourier transform of the
    gridded visibilities.

    Parameters
    ----------
    grid_data : pyvisgrid.GridData
        The gridded data from the ``pyvisgrid.Gridder.grid`` method.
        This always represents the gridded visibilities of one
        Stokes component.
    mode : str, optional
        The mode specifying which values of the mask should be plotted.
        Possible values are:

        - ``real``:     Plots the real part of the dirty image.

        - ``imag``:     Plots the imaginary part of the dirty image.

        - ``abs``:      Plot the absolute value of the dirty image.

        Default is ``real``.
    ax_unit: str | astropy.units.Unit, optional
        The unit in which to show the ticks of the x and y-axes in.
        The y-axis is the Declination (DEC) and the x-axis is the Right Ascension (RA).
        The latter one is defined as increasing from left to right!
        The unit has to be given as a string or an ``astropy.units.Unit``.
        The string must correspond to the string representation of an
        ``astropy.units.Unit``.

        Valid units are either ``pixel`` or angle units like ``arcsec``, ``degree``
        etc. Default is ``pixel``.
    center_pos: tuple | None, optional
        The coordinate center of the image. The coordinates have to
        be given in the unit defined in the parameter ``ax_unit`` above.
        If ``ax_unit`` is set to ``pixel`` this parameter is ignored.
        Default is ``None``, meaning the coordinates of the axes will be
        given as relative.
    norm : str | matplotlib.colors.Normalize | None, optional
        The name of the norm or a matplotlib norm.
        Possible string values are:

        - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                            values above the maximum will be mapped to the maximum and
                            values below the minimum will be mapped to the minimum, thus
                            avoiding the appearance of a colormaps 'over' and 'under'
                            colors (e.g. in case of negative values). Depending on the
                            use case this is desirable but in case that it is not, one
                            can set the norm to ``log_noclip`` or provide a custom norm.

        - ``log_noclip``:   Returns a logarithmic norm with clipping off.

        - ``centered``:     Returns a linear norm which centered around zero.

        - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                            square-root of the values.

        - other:            A value not declared above will be returned as is, meaning
                            that this could be any value which exists in matplotlib
                            itself.

        Default is ``None``, meaning no norm will be applied.
    colorbar_shrink: float, optional
        The shrink parameter of the colorbar. This can be needed if the plot is
        included as a subplot to adjust the size of the colorbar.
        Default is ``1``, meaning original scale.
    cmap: str | matplotlib.colors.Colormap, optional
        The colormap to be used for the plot.
        Default is ``'inferno'``.
    plot_args : dict, optional
        The additional arguments passed to the scatter plot.
        Default is ``{"color":"royalblue"}``.
    fig_args : dict, optional
        The additional arguments passed to the figure.
        If a figure object is given in the ``fig`` parameter, this
        value will be discarded.
        Default is ``{}``.
    save_to : str | None, optional
        The name of the file to save the plot to.
        Default is ``None``, meaning the plot won't be saved.
    save_args : dict, optional
        The additional arguments passed to the ``fig.savefig`` call.
        Default is ``{"bbox_inches":"tight"}``.
    fig : matplotlib.figure.Figure | None, optional
        A custom figure object.
        If set to ``None``, the ``ax`` parameter also has to be ``None``!
        Default is ``None``.
    ax : matplotlib.axes.Axes | None, optional
        A custom axes object.
        If set to ``None``, the ``fig`` parameter also has to be ``None``!
        Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if plot_args is None:
        plot_args = {}

    if fig_args is None:
        fig_args = {}

    if save_args is None:
        save_args = dict(bbox_inches="tight")

    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    norm = _get_norm(norm) if isinstance(norm, str) else norm

    match mode:
        case "real":
            dirty_image = grid_data.dirty_image.real
        case "imag":
            dirty_image = grid_data.dirty_image.imag
        case "abs":
            dirty_image = np.abs(grid_data.dirty_image)
        case _:
            raise ValueError(
                "The given mode does not exist! Valid modes are: real, imag, abs"
            )

    unit = units.Unit(ax_unit)

    if unit.physical_type == "angle":
        img_size = dirty_image.shape[0]
        cell_size = grid_data.fov / img_size

        extent = (
            np.array([-img_size / 2, img_size / 2] * 2) * cell_size * units.rad
        ).to(unit)

        if center_pos is not None:
            center_pos = (np.array(center_pos) * units.degree).to(unit)
            extent[:2] += center_pos[0]
            extent[2:] += center_pos[1]
            label_prefix = ""
        else:
            label_prefix = "Relative "

        ax.set_xlabel(f"{label_prefix}RA / {unit}")
        ax.set_ylabel(f"{label_prefix}DEC / {unit}")

        extent = extent.value

    else:
        if unit != units.pixel:
            warnings.warn(
                f"The given unit {unit} is no angle unit! Using pixels instead.",
                stacklevel=2,
            )

        extent = None

        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")

    im = ax.imshow(
        dirty_image,
        norm=norm,
        origin="lower",
        interpolation="none",
        cmap=cmap,
        extent=extent,
        **plot_args,
    )

    fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Flux Density / Jy/pix")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return fig, ax
