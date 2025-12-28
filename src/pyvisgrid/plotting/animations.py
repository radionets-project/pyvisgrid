import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
from astropy import units
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
from cartopy.feature.nightshade import Nightshade
from radiotools.layouts import Layout

from pyvisgrid.plotting import _configure_axes

__all__ = ["plot_earth_layout"]

_default_colors = mpl.colormaps["inferno"].resampled(10).colors


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
    coastline_width: float = 0.7,
    show_terrain_texture: bool = True,
    show_grid_lines: bool = True,
    show_night_shade: bool = True,
    plot_colors: dict | None = None,
    fig_args: dict | None = None,
    fig: mpl.figure.Figure | None = None,
    mosaic_axes: dict[mpl.axes.Axes] | None = None,
    mosaic_axes_key: str = "earth",
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)

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
        color=colors[5],
        s=15,
        label="Antenna positions",
        zorder=2,
    )
    ax.scatter(
        x=src_pos_itrs.lon.deg,
        y=src_pos_itrs.lat.deg,
        color=colors[-2],
        s=200,
        label="Projected source position",
        zorder=3,
        marker="*",
    )

    projection._threshold *= 100
    ax.plot(
        connection_vecs.lon,
        connection_vecs.lat,
        transform=transform,
        color=colors[0],
        linewidth=0.5,
        zorder=1,
    )
    projection._threshold = threshold_original

    return fig, ax, mosaic_axes


#
# def plot_observation_state(
#     show_masks: bool = True,
#     mask_mode: str = "amp_phase",
#     show_axes_ticks: bool = False,
#     axes_options: dict | None = None,
# ):
#     pass
