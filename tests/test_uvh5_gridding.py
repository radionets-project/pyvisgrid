from pathlib import Path

import h5py
import numpy as np
import pytest

from pyvisgrid import Gridder

def random_data(rng, size=100, cmplx=True):
    if cmplx:
        return (rng.random(size) + 1j * rng.random(size)).reshape(-1, 1)

    return rng.random(size)

@pytest.fixture
def mock_data(tmp_path) -> Path:
    rng = np.random.default_rng(42)

    output_file: Path = tmp_path / "test_data.uvh5"

    with h5py.File(output_file, "w") as f:
        vis_grp = f.create_group("visibilities")
        vis_grp.create_dataset("V_11", data=random_data(rng))
        vis_grp.create_dataset("V_22", data=random_data(rng))
        vis_grp.create_dataset("V_12", data=random_data(rng))
        vis_grp.create_dataset("V_21", data=random_data(rng))

        uvw_grp = f.create_group("uvw")
        uvw_grp.create_dataset("u", data=random_data(rng, cmplx=False))
        uvw_grp.create_dataset("v", data=random_data(rng, cmplx=False))
        uvw_grp.create_dataset(
            "st_id_pairs",
            data=np.array(np.meshgrid(np.arange(10), np.arange(10))).T.reshape(-1,2)
        )

        freq_bands = np.array([15e9])
        f.create_dataset("frequency_bands", data=freq_bands)

        sky_grp = f.create_group("sky")
        sky_grp.create_dataset("SI", data=rng.random((64, 64)))

    return output_file

@pytest.fixture
def mock_data_no_sky(tmp_path) -> Path:
    rng = np.random.default_rng(42)

    output_file: Path = tmp_path / "test_data.uvh5"

    with h5py.File(output_file, "w") as f:
        vis_grp = f.create_group("visibilities")
        vis_grp.create_dataset("V_11", data=random_data(rng))
        vis_grp.create_dataset("V_22", data=random_data(rng))
        vis_grp.create_dataset("V_12", data=random_data(rng))
        vis_grp.create_dataset("V_21", data=random_data(rng))

        uvw_grp = f.create_group("uvw")
        uvw_grp.create_dataset("u", data=random_data(rng, cmplx=False))
        uvw_grp.create_dataset("v", data=random_data(rng, cmplx=False))
        uvw_grp.create_dataset(
            "st_id_pairs",
            data=np.array(np.meshgrid(np.arange(10), np.arange(10))).T.reshape(-1,2)
        )

        freq_bands = np.array([15e9])
        f.create_dataset("frequency_bands", data=freq_bands)

    return output_file


class TestUVH5Gridding:
    @pytest.mark.parametrize(
        "fov", ([0.024, 0.1, 10, 100])
    )
    def test_defaults(self, fov, mock_data) -> None:
        file = mock_data

        gridder = Gridder.from_uvh5(file, fov=fov)
        gridder.grid("I")

        assert len(gridder["I"].vis_data) == 100
        assert gridder["I"].mask is not None
        assert gridder["I"].mask_real is not None
        assert gridder["I"].mask_imag is not None
        assert gridder["I"].dirty_image is not None


    @pytest.mark.parametrize(
        "img_size", ([64, 128, 256, 512])
    )
    def test_img_size(self, img_size, mock_data) -> None:
        file = mock_data

        gridder = Gridder.from_uvh5(file, fov=0.024, img_size=img_size)
        gridder.grid("I")

        assert gridder["I"].mask is not None
        assert gridder["I"].mask_real is not None
        assert gridder["I"].mask_imag is not None
        assert gridder["I"].dirty_image is not None
        assert gridder["I"].dirty_image.shape == (img_size, img_size)

    @pytest.mark.parametrize(
        "vis_shape",
        (
            [
                100,
                (100, 1),
                (100, 1, 1, 1),
                (100, 1, 1, 1, 1, 1, 1, 1)
            ]
        )
    )
    def test_raises_vis_shape(self, vis_shape,mock_data, mocker) -> None:
        rng = np.random.default_rng(42)
        file = mock_data

        mock_np_permute_dims= mocker.patch(
            "pyvisgrid.core.gridder.np.permute_dims",
            return_value=rng.random(vis_shape),
        )

        with pytest.raises(RuntimeError) as excinfo:
            gridder = Gridder.from_uvh5(file, fov=0.024)

        assert "Expected vis_data to be of dimension 3 or 7" in str(excinfo.value)


    def test_raises_img_size(self, mock_data_no_sky) -> None:
        file = mock_data_no_sky

        with pytest.raises(RuntimeError) as excinfo:
            gridder = Gridder.from_uvh5(file, fov=0.024)

        assert f"'img_size' could not be read from {file}" in str(excinfo.value)

    @pytest.mark.parametrize(
        "station_ids_unavail", ([0.1, 0.5, 0.75, 0.9])
    )
    def test_reduce_array(self, station_ids_unavail, mock_data) -> None:
        file = mock_data

        full_gridder = Gridder.from_uvh5(
            file,
            fov=0.024,
            station_ids_unavail=0.0
        )
        full_gridder.grid("I")

        gridder = Gridder.from_uvh5(
            file,
            fov=0.024,
            station_ids_unavail=station_ids_unavail
        )
        gridder.grid("I")

        assert gridder["I"].mask is not None
        assert gridder["I"].mask_real is not None
        assert gridder["I"].mask_imag is not None
        assert gridder["I"].dirty_image is not None

        assert len(gridder["I"].vis_data) < len(full_gridder["I"].vis_data)


