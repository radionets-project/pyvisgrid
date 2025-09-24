def test_fits_gridding_SI():
    import numpy as np
    from astropy.io import fits

    from pyvisgrid import Gridder

    model_fits = fits.open("./tests/data/test_model.fits")[0]
    model_img = model_fits.data[0, 0]

    img_size = model_img.shape[0]
    fov = np.abs(model_fits.header["CDELT1"] * 3600) * img_size

    gridder = Gridder.from_fits(
        path="./tests/data/test_vis_dense.fits", img_size=img_size, fov=fov
    )
    gridder.grid("I")

    assert gridder["I"].mask is not None
    assert gridder["I"].mask_real is not None
    assert gridder["I"].mask_imag is not None
    assert gridder["I"].dirty_image is not None

    assert gridder["I"].dirty_image.shape == model_img.shape

    assert np.allclose(gridder["I"].dirty_image.real, model_img)
