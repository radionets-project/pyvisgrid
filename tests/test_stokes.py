import h5py
import numpy as np
import pytest

from pyvisgrid import Gridder
from pyvisgrid.core.stokes import get_stokes_from_vis_data

# Stokes values of two visibilities
I = np.array([3.0, 1.5])  # noqa: E741
Q = np.array([0.5, -0.2])
U = np.array([0.2, 0.4])
V = np.array([0.1, -0.3])


def pyvisgen_vis(v11, v22, v12, v21) -> np.ndarray:
    """Visibilities in the shape that pyvisgrid builds from the output of
    ``pyvisgen.simulation.vis_loop``: (n, 1, 1, freq, 1, correlation, re/im/weight),
    where the correlations are in the order V_11, V_22, V_12, V_21."""
    vis = np.stack([v11, v22, v12, v21], axis=-1)  # (n, correlation)
    arr = np.stack([vis.real, vis.imag, np.ones(vis.shape)], axis=-1)
    return arr[:, None, None, None, None, ...]


@pytest.fixture
def linear() -> np.ndarray:
    return pyvisgen_vis(I + Q, I - Q, U + 1j * V, U - 1j * V)


@pytest.fixture
def circular() -> np.ndarray:
    return pyvisgen_vis(I + V, I - V, Q + 1j * U, Q - 1j * U)


class TestGetStokesFromVisData:
    @pytest.mark.parametrize("polarization", ["linear", "", "anything"])
    @pytest.mark.parametrize(
        "component,expected",
        [("I", I), ("Q", Q), ("U", U), ("V", V)],
    )
    def test_linear(self, linear, polarization, component, expected) -> None:
        result = get_stokes_from_vis_data(linear, component, polarization)

        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "component,expected",
        [("I", I), ("Q", Q), ("U", U), ("V", V)],
    )
    def test_circular(self, circular, component, expected) -> None:
        result = get_stokes_from_vis_data(circular, component, "circular")

        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "component,expected",
        [
            ("I+Q", I + Q),
            ("I-Q", I - Q),
            ("U+V", U + 1j * V),
            ("U-V", U - 1j * V),
        ],
    )
    def test_linear_correlations(self, linear, component, expected) -> None:
        result = get_stokes_from_vis_data(linear, component, "linear")

        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "component,expected",
        [
            ("I+V", I + V),
            ("I-V", I - V),
            ("Q+U", Q + 1j * U),
            ("Q-U", Q - 1j * U),
        ],
    )
    def test_circular_correlations(self, circular, component, expected) -> None:
        result = get_stokes_from_vis_data(circular, component, "circular")

        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize("polarization", ["linear", "circular", ""])
    def test_unpolarised_source(self, polarization) -> None:
        """In the standard convention, an unpolarised source has V_11 = V_22 = I."""
        zeros = np.zeros_like(I)
        vis = pyvisgen_vis(I + zeros, I + zeros, zeros + 0j, zeros + 0j)

        result = get_stokes_from_vis_data(vis, "I", polarization)

        np.testing.assert_allclose(result, I, atol=1e-12)


def write_uvh5(path, v11, v22, normalize=None) -> None:
    """Minimal UVH5 file in the layout of pyvisgen."""
    n = len(v11)

    with h5py.File(path, "w") as f:
        vis = f.create_group("visibilities")
        vis.create_dataset("V_11", data=v11.reshape(-1, 1))
        vis.create_dataset("V_22", data=v22.reshape(-1, 1))
        vis.create_dataset("V_12", data=np.zeros((n, 1), dtype=complex))
        vis.create_dataset("V_21", data=np.zeros((n, 1), dtype=complex))

        uvw = f.create_group("uvw")
        uvw.create_dataset("u", data=np.linspace(-1e3, 1e3, n))
        uvw.create_dataset("v", data=np.linspace(1e3, -1e3, n))
        uvw.create_dataset("st_id_pairs", data=np.zeros((n, 2), dtype=int))

        f.create_dataset("frequency_bands", data=np.array([15e9]))
        f.create_dataset("times", data=np.arange(n))

        obs = f.create_group("obs")
        obs.create_dataset("ra", data=np.array([10.0]))
        obs.create_dataset("dec", data=np.array([20.0]))

        if normalize is not None:
            f.create_dataset("normalize", data=np.bool_(normalize))


class TestFromUvh5:
    @pytest.fixture
    def flux(self) -> np.ndarray:
        return np.linspace(1.0, 2.0, 20) * (1 + 0.5j)

    def stokes_i(self, path) -> np.ndarray:
        gridder = Gridder.from_uvh5(path, fov=100.0, img_size=64)

        return np.asarray(gridder.stokes["I"].vis_data)

    def test_standard_convention(self, tmp_path, flux) -> None:
        path = tmp_path / "standard.uvh5"
        write_uvh5(path, flux, flux, normalize=False)

        np.testing.assert_allclose(self.stokes_i(path).ravel(), flux)

    def test_normalized_data_scaled_back(self, tmp_path, flux) -> None:
        """Visibilities that were simulated with normalize=True hold I / 2."""
        path = tmp_path / "normalized.uvh5"
        write_uvh5(path, 0.5 * flux, 0.5 * flux, normalize=True)

        np.testing.assert_allclose(self.stokes_i(path).ravel(), flux)

    def test_without_normalize_flag_is_standard(self, tmp_path, flux) -> None:
        path = tmp_path / "no_flag.uvh5"
        write_uvh5(path, flux, flux, normalize=None)

        np.testing.assert_allclose(self.stokes_i(path).ravel(), flux)


class TestFromPyvisgen:
    """Gridded Stokes I has to be the Fourier transform of the sky, independent
    of the normalization of the simulation."""

    @pytest.fixture
    def simulate(self):
        pytest.importorskip("pyvisgen")
        import datetime

        import torch
        from pyvisgen.simulation import Observation, vis_loop

        size = 32
        obs = Observation(
            src_ra=180.0,
            src_dec=-35.0,
            start_time=datetime.datetime(2026, 1, 21),
            scan_duration=400,
            num_scans=1,
            scan_separation=0,
            integration_time=60,
            ref_frequency=1.284e9,
            frequency_offsets=[0.0],
            bandwidths=[1e8],
            fov=0.1,
            image_size=size,
            array_layout="meerkat_test",
            corrupted=False,
            device="cpu",
        )
        sky = torch.zeros(1, size, size)
        sky[0, size // 2, size // 2] = 0.010  # 10 mJy at the phase centre

        def run(normalize: bool):
            vis = vis_loop(obs, sky, noise_level=0, normalize=normalize)

            return vis, obs

        return run

    @pytest.mark.parametrize("normalize", [True, False])
    def test_stokes_i_is_flux_of_source(self, simulate, normalize) -> None:
        vis, obs = simulate(normalize)

        gridder = Gridder.from_pyvisgen(
            vis_data=vis,
            obs=obs,
            img_size=64,
            fov=0.1,
            stokes_components="I",
            polarizations=None,
            normalized=normalize,
        )

        amplitude = np.abs(np.asarray(gridder.stokes["I"].vis_data))
        np.testing.assert_allclose(amplitude, 0.010, rtol=1e-3)
