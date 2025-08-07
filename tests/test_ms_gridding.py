def test_ms_gridding_SI():
    import shutil

    import h5py
    import numpy as np
    from astropy.io import fits

    from pyvisgrid import Gridder

    model_fits = fits.open("./tests/data/test_model.fits")[0]
    model_img = model_fits.data[0, 0]

    img_size = model_img.shape[0]
    fov = np.abs(model_fits.header["CDELT1"] * 3600) * img_size

    shutil.unpack_archive(
        "./tests/data/test_vis.ms.zip", extract_dir="./tests/data/test_vis_temp.ms"
    )

    gridder = Gridder.from_ms(
        path="./tests/data/test_vis_temp.ms", img_size=img_size, fov=fov
    )

    shutil.rmtree("./tests/data/test_vis_temp.ms")

    gridder.grid("I")

    assert gridder["I"].mask is not None
    assert gridder["I"].mask_real is not None
    assert gridder["I"].mask_imag is not None
    assert gridder["I"].dirty_image is not None

    assert gridder["I"].dirty_image.shape == model_img.shape

    with h5py.File("./tests/data/test_vis_grid_ms.h5", "r") as hf:
        assert np.allclose(gridder["I"].mask, hf["mask_hist"][()])
        assert np.allclose(
            gridder["I"].mask_real + 1j * gridder["I"].mask_imag, hf["mask_complex"][()]
        )
        assert np.allclose(gridder["I"].dirty_image.real, hf["dirty_image"][()].real)
