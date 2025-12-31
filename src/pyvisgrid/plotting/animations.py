from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
from cartopy.feature.nightshade import Nightshade
from matplotlib.ticker import NullFormatter
from mergedeep import merge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from radiotools.layouts import Layout
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from pyvisgrid.core.gridder import GridData, GridDataSeries

from pyvisgrid.plotting.plotting import _configure_axes, _get_norm

__all__ = ["plot_earth_layout", "plot_observation_state", "animate_observation"]

_default_colors = mpl.colormaps["inferno"].resampled(10).colors


def _is_value_in(value: object, lst: list):
    return value in np.ravel(lst)


# based on https://stackoverflow.com/a/18195921 by "bogatron"
def _configure_colorbar(
    mappable: mpl.cm.ScalarMappable,
    ax: mpl.axes.Axes,
    fig: mpl.figure.Figure,
    label: str | None,
    show_ticks: bool,
    fontsize: str = "medium",
) -> mpl.colorbar.Colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=fontsize)

    if not show_ticks:
        cbar.set_ticks([])
        cbar.ax.yaxis.set_major_formatter(NullFormatter())
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        cbar.ax.tick_params(labelsize=fontsize)

    return cbar


def _times2hours(times: np.ndarray):
    times = Time(times, format="mjd")
    times = times.unix
    times -= times[0]
    times /= 3600

    return times


def plot_earth_layout(
    layout: Layout | str,
    src_ra: float,
    src_dec: float,
    current_time: Time,
    show_legend: bool = True,
    legend_args: dict | None = None,
    legend_fontsize: str | int = "x-small",
    show_title: bool = True,
    title_fontsize: str | int = "small",
    coastline_width: float = 0.7,
    show_terrain_texture: bool = True,
    show_grid_lines: bool = True,
    show_night_shade: bool = True,
    marker_sizes: dict | None = None,
    plot_colors: dict | None = None,
    fig_args: dict | None = None,
    fig: mpl.figure.Figure | None = None,
    mosaic_axes: dict[mpl.axes.Axes] | None = None,
    mosaic_axes_key: str = "earth",
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    if marker_sizes is None:
        marker_sizes = {
            "antennas": 13,
            "source": 150,
        }

    if plot_colors is None:
        colors = {
            "antennas": _default_colors[5],
            "source": _default_colors[-2],
            "connections": _default_colors[0],
        }

    if isinstance(layout, str):
        layout = Layout.from_pyvisgen(layout)

    src_pos = SkyCoord(
        ra=src_ra,
        dec=src_dec,
        unit=(units.deg, units.deg),
    )

    src_pos_itrs = src_pos.transform_to(ITRS(obstime=current_time)).spherical

    projection = ccrs.NearsidePerspective(
        central_longitude=src_pos_itrs.lon,
        central_latitude=src_pos_itrs.lat,
        satellite_height=1e10,
    )

    if mosaic_axes is not None:
        gs = ax.get_subplotspec()
        fig.delaxes(ax)
        ax = fig.add_subplot(gs, projection=projection)
        mosaic_axes[mosaic_axes_key] = ax

    threshold_original = projection._threshold

    if show_title:
        ax.set_title(current_time.iso[:-4], fontsize=title_fontsize)

    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black")
    if show_night_shade:
        ax.add_feature(Nightshade(date=current_time.to_datetime(), alpha=0.2))

    ax.set_global()
    if show_grid_lines:
        ax.gridlines()
    if show_terrain_texture:
        ax.stock_img()

    ax.coastlines(linewidth=coastline_width)

    transform = ccrs.Geodetic()

    antennas = layout.get_antenna_positions()
    connection_vecs = layout.get_station_combinations()

    ax.scatter(
        x=antennas.lon,
        y=antennas.lat,
        transform=transform,
        color=colors["antennas"],
        s=marker_sizes["antennas"],
        label="Antenna positions",
        zorder=2,
    )
    ax.scatter(
        x=src_pos_itrs.lon.deg,
        y=src_pos_itrs.lat.deg,
        color=colors["source"],
        s=marker_sizes["source"],
        label="Projected source position",
        zorder=3,
        marker="*",
    )

    if show_legend:
        if legend_args is None:
            legend_args = {
                "loc": "center",
                "bbox_to_anchor": (0.5, -0.12),
                "fontsize": legend_fontsize,
                "borderaxespad": 0,
            }

        ax.legend(**legend_args)

    projection._threshold *= 100
    ax.plot(
        connection_vecs.lon,
        connection_vecs.lat,
        transform=transform,
        color=colors["connections"],
        linewidth=0.5,
        zorder=1,
    )
    projection._threshold = threshold_original

    return fig, ax, mosaic_axes


def plot_observation_state(
    vis_data: GridData,
    u: np.ndarray,
    v: np.ndarray,
    times: np.ndarray,
    src_ra: float,
    src_dec: float,
    layout: Layout | str,
    max_values: tuple[GridData, np.ndarray, np.ndarray, np.ndarray] | None = None,
    uv_max_extension: float = 0.2,
    plot_positions: list[list[str]] | None = None,
    dirty_image_mode: str = "real",
    mask_mode: str = "amp_phase",
    swap_masks: bool = False,
    axes_options: dict | None = None,
):
    if axes_options is None:
        axes_options = {}

    if plot_positions is None:
        plot_positions = [["mask_hi", "earth", "uv"], ["mask_lo", "earth", "di"]]

    if mask_mode == "amp_phase":
        mask_cmaps = ("viridis", "RdBu")
        mask_labels = ("Amplitude / a.u.", "Phase / rad")
        mask_norms = ("log", None)
    elif mask_mode == "real_imag":
        mask_cmaps = ("PiYG", "RdBu")
        mask_labels = ("Real part / a.u.", "Imaginary part / a.u.")
        mask_norms = ("centered", "centered")
    else:
        raise ValueError("Possible mask_modes: 'amp_phase' or 'real_imag'")

    if swap_masks:
        mask_cmaps = mask_cmaps[::-1]
        mask_labels = mask_labels[::-1]
        mask_norms = mask_norms[::-1]

    default_axes_options = {
        "uv": {
            "show_title": True,
            "title_fontsize": "medium",
            "axes_ticks": False,
            "axes_labels": True,
            "axes_fontsize": "x-small",
            "show_times": True,
            "cmap": "viridis",
            "show_cbar": True,
            "cbar_ticks": True,
            "cbar_label": True,
            "cbar_fontsize": "small",
            "color": _default_colors[4],
            "aspect": "equal",
        },
        "di": {
            "show_title": True,
            "title_fontsize": "medium",
            "axes_ticks": False,
            "axes_labels": False,
            "axes_fontsize": "x-small",
            "cmap": "inferno",
            "norm": "sqrt",
            "show_cbar": True,
            "cbar_ticks": False,
            "cbar_label": True,
            "cbar_fontsize": "small",
            "mode_in_label": True,
        },
        "mask_hi": {
            "show_title": True,
            "title_fontsize": "medium",
            "axes_ticks": False,
            "axes_labels": False,
            "axes_fontsize": "x-small",
            "cmap": mask_cmaps[0],
            "label": mask_labels[0],
            "norm": mask_norms[0],
            "show_cbar": True,
            "cbar_ticks": False,
            "cbar_label": True,
            "cbar_fontsize": "small",
        },
        "mask_lo": {
            "show_title": True,
            "title_fontsize": "medium",
            "axes_ticks": False,
            "axes_labels": False,
            "axes_fontsize": "x-small",
            "cmap": mask_cmaps[1],
            "label": mask_labels[1],
            "norm": mask_norms[1],
            "show_cbar": True,
            "cbar_ticks": False,
            "cbar_label": True,
            "cbar_fontsize": "small",
        },
        "earth": {
            "show_title": True,
            "title_fontsize": "small",
            "show_legend": True,
            "legend_args": None,
            "legend_fontsize": "x-small",
            "coastline_width": 0.7,
            "show_terrain_texture": True,
            "show_grid_lines": True,
            "show_night_shade": True,
            "marker_sizes": None,
            "plot_colors": None,
        },
    }

    axes_options = merge({}, default_axes_options, axes_options)

    fig, ax = plt.subplot_mosaic(plot_positions)

    # Set initial values

    u = np.append(-u, u)
    v = np.append(-v, v)

    # Set maximum values: max_values = [vis_data, u, v, times]

    if max_values is not None and len(max_values) != 4:
        raise ValueError(
            "The 'max_values' parameter has to have the form "
            "max_values = [vis_data, u, v, times]."
        )

    if max_values is not None:
        vis_data_max = max_values[0]
        u_max = max_values[1]
        v_max = max_values[2]
        times_max = max_values[3]
    else:
        vis_data_max = vis_data
        u_max = u
        v_max = v
        times_max = times

    u_max = np.abs(u_max).max() * (1 + uv_max_extension)
    v_max = np.abs(v_max).max() * (1 + uv_max_extension)

    # Set mask values

    if mask_mode == "amp_phase":
        mask_imgs = vis_data.get_mask_abs_phase()
        mask_imgs_max = vis_data_max.get_mask_abs_phase()
    elif mask_mode == "real_imag":
        mask_imgs = vis_data.get_mask_real_imag()
        mask_imgs_max = vis_data_max.get_mask_real_imag()

    if swap_masks:
        mask_imgs = mask_imgs[::-1]
        mask_imgs_max = mask_imgs_max[::-1]

    # Plot subplots
    if _is_value_in("uv", plot_positions):
        #
        time_hours = _times2hours(times=times)
        time_hours -= time_hours.min()

        time_hours_max = _times2hours(times=times_max)
        time_hours_max -= time_hours_max.min()

        if axes_options["uv"]["show_times"]:
            uv_scat = ax["uv"].scatter(
                x=u,
                y=v,
                c=np.tile(time_hours, reps=2),
                s=0.5,
                cmap=axes_options["uv"]["cmap"],
                vmin=time_hours_max.min(),
                vmax=time_hours_max.max(),
            )

            if axes_options["uv"]["show_cbar"]:
                _configure_colorbar(
                    mappable=uv_scat,
                    ax=ax["uv"],
                    fig=fig,
                    label="Time / h" if axes_options["uv"]["cbar_label"] else None,
                    show_ticks=axes_options["uv"]["cbar_ticks"],
                    fontsize=axes_options["uv"]["cbar_fontsize"],
                )
        else:
            uv_scat = ax["uv"].scatter(
                x=u, y=v, s=0.5, color=axes_options["uv"]["color"]
            )

        ax["uv"].set_xlim(-u_max, u_max)
        ax["uv"].set_ylim(-v_max, v_max)

        ax["uv"].set_aspect(axes_options["uv"]["aspect"])

        if axes_options["uv"]["show_title"]:
            ax["uv"].set_title(
                "Ungridded $(u,v)$", fontsize=axes_options["uv"]["title_fontsize"]
            )
        if axes_options["uv"]["axes_labels"]:
            ax["uv"].set_xlabel(
                "$u$ / $\\lambda$", fontsize=axes_options["uv"]["axes_fontsize"]
            )
            ax["uv"].set_ylabel(
                "$v$ / $\\lambda$", fontsize=axes_options["uv"]["axes_fontsize"]
            )

        if not axes_options["uv"]["axes_ticks"]:
            ax["uv"].set_xticks([])
            ax["uv"].set_yticks([])
        else:
            ax["uv"].xaxis.set_tick_params(
                labelsize=axes_options["uv"]["axes_fontsize"]
            )
            ax["uv"].yaxis.set_tick_params(
                labelsize=axes_options["uv"]["axes_fontsize"]
            )
    else:
        uv_scat = None

    if _is_value_in("di", plot_positions):
        match dirty_image_mode:
            case "real":
                dirty_image = vis_data.dirty_image.real
            case "imag":
                dirty_image = vis_data.dirty_image.imag
            case "abs":
                dirty_image = np.abs(vis_data.dirty_image)
            case _:
                raise ValueError(
                    "The given dirty image mode does not exist! "
                    "Valid modes are: real, imag, abs"
                )

        di_im = ax["di"].imshow(
            X=dirty_image,
            cmap=axes_options["di"]["cmap"],
            norm=_get_norm(
                axes_options["di"]["norm"],
                vmin=vis_data_max.dirty_image.real[
                    vis_data_max.dirty_image.real > 0
                ].min(),
                vmax=vis_data_max.dirty_image.real.max(),
            ),
            origin="lower",
            interpolation="none",
        )

        if axes_options["di"]["show_cbar"]:
            mode_str = (
                "" if axes_options["di"]["mode_in_label"] else f" {dirty_image_mode}"
            )
            _configure_colorbar(
                mappable=di_im,
                ax=ax["di"],
                fig=fig,
                label=f"Flux density{mode_str} / Jy/pix"
                if axes_options["di"]["cbar_label"]
                else None,
                show_ticks=axes_options["di"]["cbar_ticks"],
                fontsize=axes_options["di"]["cbar_fontsize"],
            )

        if axes_options["di"]["show_title"]:
            ax["di"].set_title(
                "Dirty Image", fontsize=axes_options["di"]["title_fontsize"]
            )
        if axes_options["di"]["axes_labels"]:
            ax["di"].set_xlabel("Pixels", fontsize=axes_options["di"]["axes_fontsize"])
            ax["di"].set_ylabel("Pixels", fontsize=axes_options["di"]["axes_fontsize"])
        if not axes_options["di"]["axes_ticks"]:
            ax["di"].set_xticks([])
            ax["di"].set_yticks([])
        else:
            ax["di"].xaxis.set_tick_params(
                labelsize=axes_options["di"]["axes_fontsize"]
            )
            ax["di"].yaxis.set_tick_params(
                labelsize=axes_options["di"]["axes_fontsize"]
            )

    else:
        di_im = None

    if _is_value_in("earth", plot_positions):
        plot_earth_layout(
            layout=layout,
            src_ra=src_ra,
            src_dec=src_dec,
            current_time=Time(times[-1], format="mjd"),
            show_legend=axes_options["earth"]["show_legend"],
            legend_fontsize=axes_options["earth"]["legend_fontsize"],
            show_title=axes_options["earth"]["show_title"],
            title_fontsize=axes_options["earth"]["title_fontsize"],
            coastline_width=axes_options["earth"]["coastline_width"],
            show_terrain_texture=axes_options["earth"]["show_terrain_texture"],
            show_grid_lines=axes_options["earth"]["show_grid_lines"],
            show_night_shade=axes_options["earth"]["show_night_shade"],
            plot_colors=axes_options["earth"]["plot_colors"],
            fig=fig,
            mosaic_axes=ax,
            mosaic_axes_key="earth",
            ax=ax["earth"],
        )

    def _plot_mask(mask_img, mask_img_max, mask_key):
        if (
            mask_key == "mask_hi"
            or (mask_key == "mask_lo" and not _is_value_in("mask_hi", plot_positions))
        ) and axes_options[mask_key]["show_title"]:
            ax[mask_key].set_title(
                "Gridded Visibilities",
                fontsize=axes_options[mask_key]["title_fontsize"],
            )

        mask = ax[mask_key].imshow(
            X=mask_img,
            cmap=axes_options[mask_key]["cmap"],
            norm=_get_norm(
                axes_options[mask_key]["norm"],
                vmin=mask_img_max[mask_img_max > 0].min(),
                vmax=mask_img_max.max(),
            ),
            origin="lower",
            interpolation="none",
        )

        if axes_options[mask_key]["show_cbar"]:
            _configure_colorbar(
                mappable=mask,
                ax=ax[mask_key],
                fig=fig,
                label=axes_options[mask_key]["label"]
                if axes_options[mask_key]["cbar_label"]
                else None,
                show_ticks=axes_options[mask_key]["cbar_ticks"],
                fontsize=axes_options[mask_key]["cbar_fontsize"],
            )

        if axes_options[mask_key]["axes_labels"]:
            ax[mask_key].set_xlabel(
                "Frequels", fontsize=axes_options[mask_key]["axes_fontsize"]
            )
            ax[mask_key].set_ylabel(
                "Frequels", fontsize=axes_options[mask_key]["axes_fontsize"]
            )
        if not axes_options[mask_key]["axes_ticks"]:
            ax[mask_key].set_xticks([])
            ax[mask_key].set_yticks([])
        else:
            ax[mask_key].xaxis.set_tick_params(
                labelsize=axes_options[mask_key]["axes_fontsize"]
            )
            ax[mask_key].yaxis.set_tick_params(
                labelsize=axes_options[mask_key]["axes_fontsize"]
            )

        return mask

    plots = {
        "uv": uv_scat,
        "di": di_im,
        "earth": _is_value_in("earth", plot_positions),
        "mask_hi": None,
        "mask_lo": None,
    }

    for mask_key, mask_img, mask_img_max in zip(
        ["mask_hi", "mask_lo"], mask_imgs, mask_imgs_max
    ):
        if _is_value_in(mask_key, plot_positions):
            mask = _plot_mask(
                mask_img=mask_img, mask_img_max=mask_img_max, mask_key=mask_key
            )
            plots[mask_key] = mask

    return fig, ax, plots, axes_options


def animate_observation(
    series: GridDataSeries,
    src_ra: float,
    src_dec: float,
    layout: Layout | str,
    interval: int,
    save_to: PathLike,
    max_values: tuple[GridData, np.ndarray, np.ndarray, np.ndarray] | None = None,
    uv_max_extension: float = 0.2,
    plot_positions: list[list[str]] | None = None,
    mask_mode: str = "amp_phase",
    swap_masks: bool = False,
    dirty_image_mode: str = "real",
    axes_options: dict | None = None,
    show_progress: bool = True,
    dpi: int or str = "figure",
):
    def _progress_func(_i, _n):
        progress_bar.update(1)

    frames = len(series)

    # GridDataSeries[i] = [grid_data, u, v, times]
    init_data = series[1]
    last_data = series[-1]

    fig, ax, plots, axes_options = plot_observation_state(
        vis_data=init_data[0],
        u=init_data[1],
        v=init_data[2],
        times=init_data[3],
        src_ra=src_ra,
        src_dec=src_dec,
        layout=layout,
        max_values=[last_data[0], last_data[1], last_data[2], last_data[3]],
        uv_max_extension=uv_max_extension,
        plot_positions=plot_positions,
        mask_mode=mask_mode,
        swap_masks=swap_masks,
        axes_options=axes_options,
    )

    def update(frame):
        return_vals = []
        for val in plots.values():
            if val is not None and not isinstance(val, bool):
                return_vals.append(val)

        if frame == 0:
            return return_vals

        vis_data, u, v, times = series[frame]

        # Update uv plot

        uv_scat = plots["uv"]

        if uv_scat is not None:
            u = np.append(-u, u)
            v = np.append(-v, v)
            uv_scat.set_offsets(np.stack([u, v]).T)
            uv_scat.set_array(np.tile(_times2hours(times=times), reps=2))

        # Update dirty image

        di_im = plots["di"]

        if di_im is not None:
            match dirty_image_mode:
                case "real":
                    dirty_image = vis_data.dirty_image.real
                case "imag":
                    dirty_image = vis_data.dirty_image.imag
                case "abs":
                    dirty_image = np.abs(vis_data.dirty_image)

            di_im.set_data(dirty_image)

        # Update masks

        if mask_mode == "amp_phase":
            mask_imgs = vis_data.get_mask_abs_phase()
        elif mask_mode == "real_imag":
            mask_imgs = vis_data.get_mask_real_imag()

        if swap_masks:
            mask_imgs = mask_imgs[::-1]

        mask_hi = plots["mask_hi"]

        if mask_hi is not None:
            mask_hi.set_data(mask_imgs[0])

        mask_lo = plots["mask_lo"]

        if mask_lo is not None:
            mask_lo.set_data(mask_imgs[1])

        # Update earth

        if plots["earth"]:
            current_time = Time(times[-1], format="mjd")
            plot_earth_layout(
                layout=layout,
                src_ra=src_ra,
                src_dec=src_dec,
                current_time=current_time,
                show_legend=axes_options["earth"]["show_legend"],
                show_title=axes_options["earth"]["show_title"],
                coastline_width=axes_options["earth"]["coastline_width"],
                show_terrain_texture=axes_options["earth"]["show_terrain_texture"],
                show_grid_lines=axes_options["earth"]["show_grid_lines"],
                show_night_shade=axes_options["earth"]["show_night_shade"],
                plot_colors=axes_options["earth"]["plot_colors"],
                fig=fig,
                mosaic_axes=ax,
                mosaic_axes_key="earth",
                ax=ax["earth"],
            )

        return return_vals

    if isinstance(save_to, str):
        save_to = Path(save_to)

    writer = None
    if save_to.suffix.lower() == ".gif":
        writer = animation.PillowWriter(
            fps=1 / (interval * 1e-3),
            bitrate=-1,
        )
        writer.setup(fig=fig, outfile=save_to, dpi=dpi)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, blit=False, interval=interval
    )

    with tqdm(
        total=frames, desc="Saving animation", disable=not show_progress
    ) as progress_bar:
        if writer is None:
            ani.save(save_to, progress_callback=_progress_func, dpi=dpi)
        else:
            ani.save(save_to, progress_callback=_progress_func, writer=writer, dpi=dpi)
