from dataclasses import dataclass
from pathlib import Path

import numpy
import numpy as np
from astropy.constants import c
from astropy.io import fits
from casatools.table import table


@dataclass
class GridData:
    """
    DataClass to save the gridded and non-gridded visibilities for a
    specific Stokes component.

    Parameters
    ----------

    vis_data : numpy.ndarray
    The ungridded visibilities.

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

    vis_data: numpy.ndarray
    mask: numpy.ndarray or None = None
    mask_real: numpy.ndarray or None = None
    mask_imag: numpy.ndarray or None = None
    dirty_image: numpy.ndarray or None = None

    def __str__(self):
        return self.__dict__


class Gridder:
    def __init__(
        self,
        u_meter: numpy.ndarray,
        v_meter: numpy.ndarray,
        img_size: int,
        fov: float,
        frequency: float,
    ):
        """

        Initializes the default Gridder of the radionets-project.
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

        img_size : int
        The size of the image in pixels.

        fov : float
        The physical size of the image in asec.

        frequency : float
        The reference frequency of the data in Hertz.

        """

        self.frequency = frequency

        self.u_wave = u_meter * self.frequency / c.value  # divide by wavelength
        self.v_wave = v_meter * self.frequency / c.value  # divide by wavelength

        self.u_meter = u_meter
        self.v_meter = v_meter

        self.img_size = img_size

        self.fov = np.deg2rad(fov / 3600)  # convert from asec to rad

        self.stokes = dict()  # initialize stokes component dictionary

    def __str__(self):
        return str(self.__dict__)

    def __getitem__(self, i: str):
        if not isinstance(i, str):
            raise KeyError(
                "The provided key has to be a valid Stokes component, i.e. 'I'."
            )

        return self.stokes[i]

    def grid(self, stokes_component: str = "I"):
        """
        Grids given visibility data using the default Gridder for the
        radionets-project.

        Parameters
        ----------

        stokes_component : str, optional
        The symbol of the stokes component which should be gridded.
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

        grid_data = self.stokes[stokes_component]

        stokes = grid_data.vis_data

        real = stokes.real.T
        imag = stokes.imag.T

        u_wave_full = np.append(-self.u_wave.ravel(), self.u_wave.ravel())
        v_wave_full = np.append(-self.v_wave.ravel(), self.v_wave.ravel())
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
            u_wave_full, v_wave_full, bins=[bins, bins], density=False
        )
        mask[mask == 0] = 1

        mask_real, _, _ = np.histogram2d(
            u_wave_full,
            v_wave_full,
            bins=[bins, bins],
            weights=stokes_real_full,
            density=False,
        )
        mask_imag, _, _ = np.histogram2d(
            u_wave_full,
            v_wave_full,
            bins=[bins, bins],
            weights=stokes_imag_full,
            density=False,
        )
        mask_real /= mask
        mask_imag /= mask

        grid_data.mask = mask
        grid_data.mask_real = mask_real
        grid_data.mask_imag = mask_imag
        grid_data.dirty_image = np.fft.fftshift(
            np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag))
        )

        return self[stokes_component]

    @classmethod
    def from_fits(cls, path: str, img_size: int, fov: float):
        """

        Initializes the gridder with the visibility data in a given FITS file
        using the default Gridder for the radionets-project.
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

        frequency : float
        The reference frequency of the data in Hertz.

        """

        path = Path(path)

        if not path.is_file() or path.suffix.lower() != ".fits":
            raise FileNotFoundError(
                f"The file {path} is not valid! You have to select a valid .fits file!"
            )

        file = fits.open(path)

        data = file[0].data.T

        u_meter = data["UU"].T * c.value
        v_meter = data["VV"].T * c.value

        vis = file[0].data["DATA"]
        stokes_i = (
            (vis[..., 0, 0] + 1j * vis[..., 0, 1])
            + (vis[..., 1, 0] + 1j * vis[..., 1, 1])
        ).ravel()[:, None]

        cls = cls(
            u_meter=u_meter,
            v_meter=v_meter,
            img_size=img_size,
            fov=fov,
            frequency=file[0].header["CRVAL4"],
        )

        cls.stokes["I"] = GridData(vis_data=stokes_i)

        return cls

    @classmethod
    def from_ms(
        cls,
        path: str,
        img_size: int,
        fov: float,
        desc_id: int or None = None,
        fallback_frequency: float = 230e9,
    ):
        """
        Initializes the Gridder with a measurement which is saved in an
        NRAO CASA Measurement Set Currently only extraction of the Stokes I
        component is supported.

        Parameters
        ----------

        path: str
        The path of the measurement set root directory.

        img_size: int
        The size of the image in pixels.

        fov: float
        The physical size of the image in asec.

        desc_id: int, optional
        The desc_id of the visibilites which should be gridded.
        This can be used to choose the component of a composite observation.
        Default is ``None``, which means that all observations will be used.

        fallback_frequency: float, optional
        The reference frequency in Hertz that will be used, in case there is no
        clear reference frequency present in the Measurement Set. Default is ``230e9``.

        """

        path = Path(path)

        if not path.is_dir():
            raise NotADirectoryError(
                f"This measurement set does not exist under the path {path}"
            )

        tab = table(str(path))

        if desc_id is not None:
            mask = tab.getcol("DATA_DESC_ID") == desc_id
            data = tab.getcol("DATA")[:, :, mask].T
            uvw = tab.getcol("UVW")[:, mask].T
        else:
            data = tab.getcol("DATA").T
            uvw = tab.getcol("UVW").T

        try:
            freq = table(str(path / "SPECTRAL_WINDOW")).getcol("CHAN_FREQ").T
        except Exception:
            freq = fallback_frequency

        uvw = np.repeat(uvw[None], 1, axis=0)
        u_meter = uvw[:, :, 0]
        v_meter = uvw[:, :, 1]

        stokes_i = data[:, :, 0] + data[:, :, 1]

        cls = cls(
            u_meter=u_meter,
            v_meter=v_meter,
            img_size=img_size,
            fov=fov,
            frequency=freq,
        )

        cls.stokes["I"] = GridData(vis_data=stokes_i)

        return cls
