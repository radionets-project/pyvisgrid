from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time
from casacore.tables import table
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import matplotlib
    from pyvisgen.simulation import Observation, Visibilities
    from radiotools.layouts import Layout

try:
    import pandas as pd
    from radiotools.layouts import Layout

    include_array_layout = True
except ImportError:
    include_array_layout = False


import pyvisgrid.plotting as plotting
from pyvisgrid.core.stokes import get_stokes_from_vis_data

__all__ = ["GridData", "Gridder", "GridDataSeries"]


@dataclass
class GridData:
    """DataClass to save the gridded and non-gridded visibilities for a
    specific Stokes component.

    Attributes
    ----------
    vis_data : numpy.ndarray
        The ungridded visibilities of shape ``(N_MEASUREMENTS * N_CHANNELS,)``.

    fov : float
        The size of the Field Of View of the gridded data in arcseconds.

    mask : numpy.ndarray, optional
        The mask created from the given (u,v) coordinates. The mask contains
        the number of (u,v) coordinates per pixel.

    mask_real : numpy.ndarray, optional
        The gridded real part of the visibilites.

    mask_imag : numpy.ndarray, optional
        The gridded imaginary part of the visibilities.

    dirty_img : numpy.ndarray, optional
        The complex Dirty Image. This is the 2-dimensional Fourier transform
        of the gridded visibilities.
    """

    vis_data: np.ndarray
    fov: float | None = None
    mask: np.ndarray | None = None
    mask_real: np.ndarray | None = None
    mask_imag: np.ndarray | None = None
    dirty_image: np.ndarray | None = None

    def __str__(self) -> str:
        return self.__dict__

    def copy(self) -> GridData:
        return GridData(
            vis_data=self.vis_data,
            fov=self.fov,
            mask=self.mask,
            mask_real=self.mask_real,
            mask_imag=self.mask_imag,
            dirty_image=self.dirty_image,
        )

    def get_mask_complex(self) -> np.ndarray:
        """Returns the gridded mask as a complex array with
        the form ``mask.real + 1j * mask.imag``.

        Returns
        -------
        numpy.ndarray
            Complex mask
        """
        return self.mask_real + 1j * self.mask_imag

    def get_mask_abs_phase(self) -> tuple[np.ndarray]:
        """Returns the gridded mask as amplitude and phase.

        Returns
        -------
        tuple[numpy.ndarray]
            Amplitude and phase of the complex mask.
        """
        mask_complex = self.get_mask_complex()
        return np.abs(mask_complex), np.angle(mask_complex)

    def get_mask_real_imag(self) -> tuple[np.ndarray]:
        """Returns the gridded mask as real and imaginary parts.

        Returns
        -------
        tuple[numpy.ndarray]
            Real and parts of the complex mask.
        """
        return self.mask_real, self.mask_imag


class GridDataSeries:
    """DataClass to save the gridded and non-gridded visibilities for a
    specific Stokes component for different time steps. This enables to
    create a time series of different steps in an observation.

    Attributes
    ----------

    grid_data : GridData
        The GridData of the full observation.

    times : numpy.ndarray
        The time steps of the observation.

    u_wave : np.ndarray
        The :math:`u` coordinates as a multiple of the wavelength.

    u_wave : np.ndarray
        The :math:`v` coordinates as a multiple of the wavelength.

    step_size : int
        The size of each timestep (how many steps in the observation are
        in each iteration of the series).

    time_cutoff_idx : int | None, optional
        The index to stop the series at.
        Default is ``None``.
    """

    def __init__(
        self,
        grid_data: GridData,
        u_wave: np.ndarray,
        v_wave: np.ndarray,
        times: np.ndarray,
        num_frequencies: int,
        step_size: int,
        time_start_idx: int = 0,
        time_end_idx: int | None = None,
    ):
        self._grid_data_full: GridData = grid_data
        self._grid_data: list[GridData] = []

        self._u_wave: np.ndarray = u_wave
        self._v_wave: np.ndarray = v_wave

        self._num_frequencies: int = num_frequencies

        self._times_full: np.ndarray = times

        self.num_steps = self._times_full.size // self._num_frequencies

        self._step_idx: np.ndarray = np.arange(self.num_steps)[
            time_start_idx:time_end_idx
        ][::step_size]
        self._times_idx: np.ndarray = np.tile(
            np.arange(0, self.num_steps),
            reps=self._num_frequencies,
        )

        self._iter_idx: int = 0

    def __len__(self) -> int:
        return self._step_idx.size

    def __getitem__(self, i) -> list[GridData, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(i, int):
            raise TypeError("The index must be an integer!")

        time_idcs = self._get_time_idcs(step=i)

        result = [self._grid_data[i]]
        result.extend(self.get_uv_step(step=i))
        result.extend([self._times_full[time_idcs]])

        return result

    def __iter__(self) -> GridDataSeries:
        self._iter_idx = 0
        return self

    def __next__(self) -> list[GridData, np.ndarray, np.ndarray, np.ndarray]:
        if self._iter_idx >= len(self):
            raise StopIteration

        num_result = self[self._iter_idx]

        self._iter_idx += 1
        return num_result

    def _get_time_idcs(self, step: int) -> np.ndarray:
        """
        Helper function which converts from animation step to time indices.

        Parameters
        ----------

        step : int
            The step index.

        Returns
        -------

        np.ndarray :
            Array of indices at which the times, coordinates and visibilities up to this
            step index are located in their respective arrays.

        """
        step_idx = self._step_idx[step]
        return np.argwhere(self._times_idx <= step_idx).ravel()

    def get_uv_step(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the ungridded :math:`(u,v)` coordinates for the given timestep.

        Parameters
        ----------
        step : int
            The step index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]:
            The :math:`(u,v)` coordinates in the order ``(u, v)``.
        """

        time_idcs = self._get_time_idcs(step=step)

        return (
            self._u_wave[time_idcs],
            self._v_wave[time_idcs],
        )

    def get_vis_step(self, step: int) -> GridData:
        """
        Returns the visibility data for the given timestep.

        Parameters
        ----------
        step : int
            The step index.

        Returns
        -------
        GridData:
            The (un)gridded visibility data.
        """

        time_idcs = self._get_time_idcs(step=step)

        grid_data = self._grid_data_full.copy()
        grid_data.vis_data = grid_data.vis_data[time_idcs]
        return grid_data

    def add_grid_data(self, grid_data: GridData) -> None:
        """
        Adds the given (un)gridded visibility data to the series.

        Parameters
        ----------

        GridData :
            The data to add to the series.
        """
        self._grid_data.append(grid_data)


class Gridder:
    def __init__(
        self,
        u_meter: np.ndarray,
        v_meter: np.ndarray,
        times: np.ndarray,
        img_size: int,
        fov: float,
        src_ra: float,
        src_dec: float,
        ref_frequency: float,
        frequency_offsets: ArrayLike,
        antenna_layout: Layout | None = None,
    ):
        """Initializes the default Gridder of the radionets-project.

        It uses NumPy histogramms to sort the
        visibilities into a grid in the Fourier space to create a
        2-dimensional mask of visibilities, which can be transformed
        into real space using a Fast Fourier Transform.

        Parameters
        ----------
        u_meter : numpy.ndarray

            The u coordinates in meter.
        v_meter : numpy.ndarray
            The v coordinates in meter.

        times : numpy.ndarray
            The times (in MJD) at which the visibilities were measured.

        img_size : int
            The size of the image in pixels.

        fov : float
            The physical size of the image in arcsec.

        src_ra : float
            The Right Ascension (RA) of the source in arcsec.

        src_dec : float
            The Declination (Dec) of the source in arcsec.

        ref_frequency : float
            The reference frequency of the data in Hertz.

        frequency_offsets : numpy.typing.ArrayLike
            The frequency offsets in Hertz.

        antenna_layout : raditools.layouts.Layout | None, optional
            The antenna layout as a radiotools ``Layout``.
            This does not need to be provided to use the Gridder but only for
            the creation of animations (needs the ``animations`` optional dependencies).
            Defaults is ``None``.
        """

        self.src_ra: float = src_ra
        self.src_dec: float = src_dec

        self.ref_frequency: float = ref_frequency
        self.frequency_offsets: np.ndarray = np.asarray(frequency_offsets).ravel()

        self.frequencies: np.ndarray = self.frequency_offsets + self.ref_frequency

        u_wave = u_meter / c.value
        v_wave = v_meter / c.value

        self.u_wave: np.ndarray = np.concatenate(
            [u_wave * freq for freq in self.frequencies]
        )
        self.v_wave: np.ndarray = np.concatenate(
            [v_wave * freq for freq in self.frequencies]
        )

        self.u_meter: np.ndarray = np.tile(u_meter, reps=self.frequencies.size)
        self.v_meter: np.ndarray = np.tile(v_meter, reps=self.frequencies.size)

        self.times: Time = Time(
            np.tile(times, reps=self.frequencies.size), format="mjd"
        )
        self.antenna_layout: Layout | None = antenna_layout

        self.img_size: int = img_size

        self.fov: float = np.deg2rad(fov / 3600)  # convert from asec to rad

        self.stokes: dict = dict()  # initialize stokes component dictionary

    def __str__(self) -> str:
        return str(self.__dict__)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, i: str) -> GridData:
        if not isinstance(i, str):
            raise KeyError(
                "The provided key has to be a valid Stokes component, i.e. 'I'."
            )

        return self.stokes[i]

    def _grid(
        self, grid_data: GridData, u_wave: np.ndarray, v_wave: np.ndarray
    ) -> GridData:
        """Internal method that grids given visibility data using the default Gridder
        for the radionets-project.

        Parameters
        ----------
        grid_data : GridData
        The ``GridData`` data object containing the visibilities.

        u_wave: numpy.ndarray
        The u coordinates in units of wavelength.

        v_wave: numpy.ndarray
        The v coordinates in units of wavelength.


        Returns
        -------
        mask : numpy.ndarray, optional
            The mask created from the given (u,v) coordinates. The mask contains
            the number of (u,v) coordinates per pixel.

        mask_real : numpy.ndarray, optional
            The gridded real part of the visibilites.

        mask_imag : numpy.ndarray, optional
            The gridded imaginary part of the visibilities.

        dirty_img_complex : numpy.ndarray, optional
            The complex Dirty Image. This is the 2-dimensional Fourier transform
            of the gridded visibilities.
        """

        stokes = grid_data.vis_data

        real = stokes.real.T
        imag = stokes.imag.T

        u_wave_full = np.append(-u_wave.ravel(), u_wave.ravel())
        v_wave_full = np.append(-v_wave.ravel(), v_wave.ravel())
        stokes_real_full = np.append(real.ravel(), real.ravel())
        stokes_imag_full = np.append(-imag.ravel(), imag.ravel())

        N = self.img_size

        delta_uv = (self.fov) ** (-1)

        bins = np.arange(
            start=-(N / 2 + 1 / 2) * delta_uv,
            stop=(N / 2 + 1 / 2) * delta_uv,
            step=delta_uv,
            dtype=np.float128,
        )

        mask, *_ = np.histogram2d(
            x=u_wave_full, y=v_wave_full, bins=[bins, bins], density=False
        )
        mask = (
            mask.T
        )  # u (x) is histogrammed along the first dimension (rows) --> transpose
        mask[mask == 0] = 1

        mask_real, _, _ = np.histogram2d(
            x=u_wave_full,
            y=v_wave_full,
            bins=[bins, bins],
            weights=stokes_real_full,
            density=False,
        )
        mask_imag, _, _ = np.histogram2d(
            x=u_wave_full,
            y=v_wave_full,
            bins=[bins, bins],
            weights=stokes_imag_full,
            density=False,
        )

        mask_real = mask_real.T  # see above
        mask_imag = mask_imag.T  # see above

        mask_real /= mask
        mask_imag /= mask

        grid_data.fov = self.fov
        grid_data.mask = mask
        grid_data.mask_real = mask_real
        grid_data.mask_imag = mask_imag
        grid_data.dirty_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag))
        )

        return grid_data

    def grid(self, stokes_component: str = "I") -> GridData:
        """Grids given visibility data using the default Gridder for the
        radionets-project.

        Parameters
        ----------
        stokes_component : str, optional
            The stokes component which should be gridded.
            The specified component has to be initialized first!
            Otherwise this will result in a ``KeyError``.
            Default is ``'I'``.

        Returns
        -------
        mask : numpy.ndarray, optional
            The mask created from the given (u,v) coordinates. The mask contains
            the number of (u,v) coordinates per pixel.

        mask_real : numpy.ndarray, optional
            The gridded real part of the visibilites.

        mask_imag : numpy.ndarray, optional
            The gridded imaginary part of the visibilities.

        dirty_img_complex : numpy.ndarray, optional
            The complex Dirty Image. This is the 2-dimensional Fourier transform
            of the gridded visibilities.
        """

        if stokes_component not in self.stokes:
            raise KeyError(
                f"The Stokes component {stokes_component} has not been initialized!"
            )

        return self._grid(
            grid_data=self.stokes[stokes_component],
            u_wave=self.u_wave,
            v_wave=self.v_wave,
        )

    def grid_time_series(
        self,
        step_size: int,
        time_start_idx: int = 0,
        time_end_idx: int | None = None,
        stokes_component: str = "I",
        show_progress: bool = True,
    ) -> GridDataSeries:
        """Grids the visibilites in a time series so that time steps get
        gridded individually.

        Parameters
        ----------

        step_size : int
            The number of visibilites (time steps) per gridded time step.
            Meaning that the resulting series will have ``len(Gridder) // step_size``
            for its length.

        time_start_idx : int, optional
            The time index at which the time series starts.
            Default is ``0``, meaning the time series starts at the first visibility.

        time_end : int | None, optional
            The time index at which the time series end.
            Default is ``None``, meaning that the time series
            ends at the last visibility.

        stokes_component : str, optional
            The Stokes component to grid. Default is ``'I'``.

        show_progress : bool, optional
            Whether to show a progress bar for the gridding progress.
            Default is ``True``.

        Returns
        -------

        GridDataSeries :
            The series of gridded and ungridded visibilities.

        """
        grid_data = self.stokes[stokes_component]
        grid_data_series = GridDataSeries(
            grid_data=grid_data,
            u_wave=self.u_wave,
            v_wave=self.v_wave,
            times=self.times.mjd,
            step_size=step_size,
            num_frequencies=self.frequencies.size,
            time_start_idx=time_start_idx,
            time_end_idx=time_end_idx,
        )

        for idx in tqdm(
            np.arange(len(grid_data_series)),
            desc="Gridding time series",
            disable=not show_progress,
        ):
            u_wave, v_wave = grid_data_series.get_uv_step(step=idx)

            grid_data_series.add_grid_data(
                grid_data=self._grid(
                    grid_data=grid_data_series.get_vis_step(step=idx),
                    u_wave=u_wave,
                    v_wave=v_wave,
                )
            )

        return grid_data_series

    @classmethod
    def from_pyvisgen(
        cls,
        vis_data: Visibilities,
        obs: Observation,
        img_size: int,
        fov: float,
        stokes_components: list[str] | str = "I",
        polarizations: list[str] | str | None = None,
    ) -> GridData:
        """Initializes the gridder with the visibility data which is generated by the
        ``pyvisgen.simulation.vis_loop`` function.
        Additionally one can define which stokes components should be calculated
        with a given polarization.

        More on the ``Gridder`` can be found in the constructor of
        the ``Gridder``.

        Parameters
        ----------
        obs : pyvisgen.simulation.Observation
            The observation which is returned by the
            ``pyvisgen.simulation.vis_loop`` function.

        vis_data : pyvisgen.simulation.Visibilities
            The visibility data which is returned by the
            ``pyvisgen.simulation.vis_loop`` function.

        img_size : int
            The size of the image in pixels.

        fov : float
            The physical size of the image in asec.

        stokes_components : list[str] | str, optional
            The Stokes components which are to be calculated and saved in the gridder.
            This can either be a list of components (e.g. ``['I', 'V']``) or a single
            string. Default is ``'I'``.

        polarizations : list[str] | str | None, optional
            The polarization type. Default is ``None``.
        """
        u_meter = vis_data.u
        v_meter = vis_data.v

        times = Time(
            obs.baselines.get_valid_subset(
                num_baselines=obs.num_baselines, device="cpu"
            ).date,
            format="jd",
        ).mjd

        vis_data = vis_data.get_values()

        if vis_data.ndim != 7:
            if vis_data.ndim == 3:
                vis_data = np.stack(
                    [vis_data.real, vis_data.imag, np.ones(vis_data.shape)],
                    axis=3,
                )[:, None, None, :, None, ...]
        else:
            raise ValueError("Expected vis_data to be of dimension 3 or 7")

        if include_array_layout:
            array = obs.array
            positions = obs.array_earth_loc.value

            layout = Layout.from_dataframe(
                df=pd.DataFrame(
                    {
                        "station_name": array.st_num,
                        "x": positions["x"],
                        "y": positions["y"],
                        "z": positions["z"],
                        "dish_dia": array.diam,
                        "el_low": array.el_low,
                        "el_high": array.el_high,
                        "sefd": array.sefd,
                        "altitude": array.altitude,
                    }
                )
            )
        else:
            layout = None

        cls = cls(
            u_meter=u_meter.cpu().numpy(),
            v_meter=v_meter.cpu().numpy(),
            times=times,
            img_size=img_size,
            fov=fov,
            src_ra=obs.ra,
            src_dec=obs.dec,
            ref_frequency=obs.ref_frequency.cpu().numpy(),
            frequency_offsets=obs.frequency_offsets,
            antenna_layout=layout,
        )

        if isinstance(stokes_components, str):
            stokes_components = [stokes_components]

        if polarizations is None:
            polarizations = ""

        if isinstance(polarizations, str):
            polarizations = [polarizations]

        if len(stokes_components) != len(polarizations):
            raise IndexError(
                "The length of stokes_components has to be equal "
                "to the length of polarizations!"
            )

        for stokes_comp, polarization in zip(stokes_components, polarizations):
            # get stokes visibilities depending on stokes component to grid
            # and polarization mode
            stokes_vis = get_stokes_from_vis_data(vis_data, stokes_comp, polarization)
            try:
                stokes_vis = stokes_vis.swapaxes(0, 1).ravel()
            except AxisError:
                stokes_vis = stokes_vis.ravel()

            # FIXME: probably some kind of difference in normalization.
            # Factor 2 fixes this for now. Has to be investigated.
            stokes_vis *= 2
            cls.stokes[stokes_comp] = GridData(vis_data=stokes_vis)

        return cls

    @classmethod
    def from_fits(
        cls,
        path: str,
        img_size: int,
        fov: float,
        uv_colnames: dict = None,
    ) -> GridData:
        """Initializes the gridder with the visibility data in a
        given FITS file using the default Gridder for the radionets-project.
        Currently only extraction of the Stokes I component is supported.
        More on the ``Gridder`` can be found in the constructor of
        the ``Gridder``.

        Parameters
        ----------
        path : str
            The path to the FITS file.

        img_size : int
            The size of the image in pixels.

        fov : float
            The physical size of the image in asec.

        uv_colnames : dict, optional
            Alternative names for the U and V columns in the FITS file.
            Default is {'u': None, 'v': None}, meaning the default values of
            'UU' and 'VV' or 'UU--' and 'VV--' will be used.
        """
        if uv_colnames is None:
            uv_colnames = dict(u=None, v=None)

        path = Path(path)
        file = fits.open(path)

        data = file[0].data.T

        if uv_colnames["u"] is None and uv_colnames["v"] is None:
            try:
                u_meter = data["UU"].T * c.value
                v_meter = data["VV"].T * c.value
            except KeyError:
                u_meter = data["UU--"].T * c.value
                v_meter = data["VV--"].T * c.value
        elif (uv_colnames["u"] is None and uv_colnames["v"] is not None) or (
            uv_colnames["u"] is not None and uv_colnames["v"] is None
        ):
            raise KeyError(
                "When providing specific column names, "
                "both the names for u and v have to be set!"
            )
        else:
            u_meter = data[uv_colnames[0]].T * c.value
            v_meter = data[uv_colnames[1]].T * c.value

        times = Time(data["DATE"], format="jd").mjd

        vis = file[0].data["DATA"]
        stokes_i = (
            (vis[..., 0, 0] + 1j * vis[..., 0, 1])
            + (vis[..., 1, 0] + 1j * vis[..., 1, 1])
        ).ravel()[:, None]

        cls = cls(
            u_meter=u_meter,
            v_meter=v_meter,
            times=times,
            img_size=img_size,
            fov=fov,
            src_ra=file[0].header["CRVAL6"] * 3600,
            src_dec=file[0].header["CRVAL7"] * 3600,
            ref_frequency=file[0].header["CRVAL4"],
            frequency_offsets=file[1].data["IF FREQ"],
            antenna_layout=Layout.from_uv_fits(path=path)
            if include_array_layout
            else None,
        )

        cls.stokes["I"] = GridData(vis_data=stokes_i)

        return cls

    @classmethod
    def from_ms(
        cls,
        path: str,
        img_size: int,
        fov: float,
        desc_id: int,
        ref_frequency_id: int = 0,
        use_calibrated: bool = False,
        filter_flagged: bool = True,
    ) -> GridData:
        """Initializes the Gridder with a measurement which is saved in an
        NRAO CASA Measurement Set. Currently only extraction of the Stokes I
        component is supported.

        Parameters
        ----------
        path: str
            The path of the measurement set root directory.

        img_size: int
            The size of the image in pixels.

        fov: float
            The physical size of the image in asec.

        desc_id: int
            The description id of the visibilites which should be gridded.
            This id corresponds to a way of choosing between the spectral windows
            of the observation and its polarization setup.

            Note: Currently it is not supported to grid all visibilities of the
            observation in one image. It is planned to add that at a later point.

        ref_frequency_id: int, optional
            The index of the reference frequency that will be used if the measurement
            is a composite observation and no desc_id is given.

        use_calibrated: bool, optional
            Whether to use the calibrated data from the MODEL_DATA column or the
            raw data from the DATA column. Default is ``True``.

        filter_flagged: bool, optional
            Whether to filter out flagged data rows. Default is ``True``.
        """
        path = Path(path)

        if not path.is_dir():
            raise NotADirectoryError(
                f"This measurement set does not exist under the path {path}"
            )

        main_tab = table(str(path), ack=False)
        spectral_tab = table(str(path / "SPECTRAL_WINDOW"), ack=False)
        source_table = table(str(path / "SOURCE"), ack=False)
        data_desc_table = table(str(path / "DATA_DESCRIPTION"), ack=False)

        spw_id = data_desc_table.getcell("SPECTRAL_WINDOW_ID", desc_id)

        data_colname = "DATA" if not use_calibrated else "MODEL_DATA"

        if desc_id is not None:
            mask = main_tab.getcol("DATA_DESC_ID") == desc_id

            main_tab = main_tab.selectrows(rownrs=np.argwhere(mask).ravel())

            ref_frequency = spectral_tab.getcell("REF_FREQUENCY", spw_id)
            frequency_offsets = (
                spectral_tab.getcell("CHAN_FREQ", spw_id) - ref_frequency
            )
        else:
            pass

        data = main_tab.getcol(data_colname)
        uv = main_tab.getcol("UVW")[:, :2]
        times = main_tab.getcol("TIME")

        if filter_flagged:
            flag_mask = main_tab.getcol("FLAG")
            flag_mask = flag_mask.reshape((flag_mask.shape[0], -1)).astype(np.uint8)
            flag_mask = np.prod(flag_mask, axis=1)

            flag_mask = np.logical_not(flag_mask.astype(bool))

        else:
            flag_mask = np.ones(uv.shape[0]).astype(bool)

        uv = uv[flag_mask]
        data = data[flag_mask]
        times = times[flag_mask]

        u_meter = uv[:, 0]
        v_meter = uv[:, 1]

        stokes_i = data[..., 0] + data[..., 1]
        stokes_i = stokes_i.T  # ensure matching shape (N_CHANNELS, N_MEASUREMENTS)

        # FIXME: probably some kind of difference in normalization.
        # Factor 0.5 fixes this for now. Has to be investigated.
        stokes_i *= 0.5

        src_ra, src_dec = np.rad2deg(source_table.getcol("DIRECTION")[0])

        cls = cls(
            u_meter=u_meter,
            v_meter=v_meter,
            times=Time(times / 3600 / 24, format="mjd").mjd,
            img_size=img_size,
            fov=fov,
            src_ra=src_ra,
            src_dec=src_dec,
            ref_frequency=ref_frequency,
            frequency_offsets=frequency_offsets,
            antenna_layout=Layout.from_measurement_set(root_path=path, sefd=0)
            if include_array_layout
            else None,
        )

        cls.stokes["I"] = GridData(vis_data=stokes_i.ravel())

        return cls

    def plot_ungridded_uv(
        self, **kwargs
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the ungridded (u,v) points as a scatter plot.

        Parameters
        ----------

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

        return plotting.plot_ungridded_uv(gridder=self, **kwargs)

    def plot_mask(
        self, stokes_component: str = "I", **kwargs
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the (u,v) mask (the binned visibilities) of the gridded
        interferometric image.

        Parameters
        ----------
        stokes_component : str, optional
            The symbol of the stokes component whose mask should be plotted.
            The specified component has to be initialized and gridded first!
            Otherwise this will result in a ``KeyError``.
            Default is ``'I'``.

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
                                values above the maximum will be mapped to the maximum
                                and values below the minimum will be mapped to the
                                minimum, thus avoiding the appearance of a colormaps
                                'over' and 'under' colors (e.g. in case of negative
                                values).
                                Depending on the use case this is desirable but in
                                case that it is not, one can set the norm to
                                ``log_noclip`` or provide a custom norm.

            - ``log_noclip``:   Returns a logarithmic norm with clipping off.

            - ``centered``:     Returns a linear norm which centered around zero.

            - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                                square-root of the values.

            - other:            A value not declared above will be returned as is,
                                meaning that this could be any value which exists in
                                matplotlib itself.

            Default is ``None``, meaning no norm will be applied.

        cmap: str | matplotlib.colors.Colormap | None, optional
            The colormap to be used for the plot.
            Default is ``None``, meaning the colormap will be default to a value
            fitting for the chosen mode.

        plot_args : dict, optional
            The additional arguments passed to the scatter plot.
            Default is ``{"color":"royalblue"}``.

        fig_args : dict | None, optional
            The additional arguments passed to the figure.
            If a figure object is given in the ``fig`` parameter, this
            value will be discarded.
            Default is ``None``.

        save_to : str | PathLike | None, optional
            The name of the file to save the plot to.
            Default is ``None``, meaning the plot won't be saved.

        save_args : dict | None, optional
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

        return plotting.plot_mask(self[stokes_component], **kwargs)

    def plot_dirty_image(
        self, stokes_component: str = "I", **kwargs
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the (u,v) dirty image, meaning the 2d Fourier transform of the
        gridded visibilities.

        Parameters
        ----------
        stokes_component : str, optional
            The symbol of the stokes component whose dirty image should be plotted.
            The specified component has to be initialized and gridded first!
            Otherwise this will result in a ``KeyError``.
            Default is ``'I'``.

        mode : str, optional
            The mode specifying which values of the dirty image should be plotted.
            Possible values are:

            - ``real``:     Plots the real part of the dirty image.

            - ``imag``:     Plots the imaginary part of the dirty image.

            - ``abs``:      Plot the absolute value of the dirty image.

            Default is ``real``.

        ax_unit: str | astropy.units.Unit, optional
            The unit in which to show the ticks of the x and y-axes in.
            The y-axis is the Declination (DEC) and the x-axis is the
            Right Ascension (RA).
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
            Possible values are:

            - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                                values above the maximum will be mapped to the maximum
                                and values below the minimum will be mapped to the
                                minimum, thus avoiding the appearance of a colormaps
                                'over' and 'under' colors (e.g. in case of negative
                                values).
                                Depending on the use case this is desirable but in
                                case that it is not, one can set the norm to
                                ``log_noclip`` or provide a custom norm.

            - ``log_noclip``:   Returns a logarithmic norm with clipping off.

            - ``centered``:     Returns a linear norm which centered around zero.

            - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                                square-root of the values.

            - other:            A value not declared above will be returned as is,
                                meaning that this could be any value which exists in
                                matplotlib itself.

            Default is ``None``, meaning no norm will be applied.

        colorbar_shrink: float, optional
            The shrink parameter of the colorbar. This can be needed if the plot is
            included as a subplot to adjust the size of the colorbar.
            Default is ``1``, meaning original scale.

        cmap: str | matplotlib.colors.Colormap, optional
            The colormap to be used for the plot.
            Default is ``'inferno'``.

        plot_args : dict | None, optional
            The additional arguments passed to the scatter plot.
            Default is ``None``.

        fig_args : dict | None, optional
            The additional arguments passed to the figure.
            If a figure object is given in the ``fig`` parameter, this
            value will be discarded.
            Default is ``None``.

        save_to : str | PathLike | None, optional
            The name of the file to save the plot to.
            Default is ``None``, meaning the plot won't be saved.

        save_args : dict | None, optional
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
        return plotting.plot_dirty_image(self[stokes_component], **kwargs)
