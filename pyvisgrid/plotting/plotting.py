import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _configure_axes(
    fig: matplotlib.figure.Figure | None,
    ax: matplotlib.axes.Axes | None,
    fig_args: dict,
):
    if None in (fig, ax) and not all(x is None for x in (fig, ax)):
        raise KeyError("The parameters ax and fig have to be both None or not None!")

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")

    return fig, ax


def _get_norm(norm: str):
    match norm:
        case "log":
            return matplotlib.colors.LogNorm(clip=True)
        case "centered":
            return matplotlib.colors.CenteredNorm()
        case "power":
            return matplotlib.colors.PowerNorm(exp=0.5)
        case _:
            return norm


def _apply_crop(ax: matplotlib.axes.Axes | None, crop: tuple[list[float | None]]):
    ax.set_xlim(crop[0][0], crop[0][1])
    ax.set_ylim(crop[1][0], crop[1][1])


def plot_ungridded_uv(
    gridder,
    mode: str = "wave",
    marker_size: float | None = None,
    plot_args: dict = dict(color="royalblue"),
    fig_args: dict = dict(),
    save_to: str | None = None,
    save_args: dict = dict(bbox_inches="tight"),
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    match mode:
        case "wave":
            u, v = gridder.u_wave, gridder.v_wave
            unit = "$1/\\lambda$"
        case "meter":
            u, v = gridder.u_meter, gridder.v_meter
            unit = "m"
        case _:
            raise ValueError(
                "The given mode does not exist! Valid modes are: wave, meter."
            )

    ax.scatter(x=np.append(-u, u), y=np.append(-v, v), s=marker_size, **plot_args)

    ax.set_xlabel(f"$u$ in {unit}")
    ax.set_ylabel(f"$v$ in {unit}")

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
    plot_args: dict = dict(),
    fig_args: dict = dict(),
    save_to: str | None = None,
    save_args: dict = dict(bbox_inches="tight"),
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
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
                im, ax=ax, shrink=colorbar_shrink, label="$(u,v)$ per frequel in 1/fq"
            )
        case "abs":
            im = ax.imshow(
                np.abs(grid_data.mask_real + 1j * grid_data.mask_imag),
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Amplitude in a.u.")
        case "phase":
            im = ax.imshow(
                np.angle(grid_data.mask_real + 1j * grid_data.mask_imag),
                norm=norm,
                origin="lower",
                interpolation="none",
                cmap=cmap,
                **plot_args,
            )
            cbar = fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Phase in rad")

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
            fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Real Part in a.u.")
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
                im, ax=ax, shrink=colorbar_shrink, label="Imaginary Part in a.u."
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
    crop: tuple[list[float | None]] = ([None, None], [None, None]),
    norm: str | matplotlib.colors.Normalize = None,
    colorbar_shrink: float = 1,
    cmap: str | matplotlib.colors.Colormap | None = "inferno",
    plot_args: dict = dict(),
    fig_args: dict = dict(),
    save_to: str | None = None,
    save_args: dict = dict(bbox_inches="tight"),
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
):
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

    im = ax.imshow(
        dirty_image,
        norm=norm,
        origin="lower",
        interpolation="none",
        cmap=cmap,
        **plot_args,
    )
    fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Flux Density in Jy/px")

    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return fig, ax
