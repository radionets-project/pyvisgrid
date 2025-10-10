===============================================================
pyvisgrid |ci| |pre-commit| |codecov| |pypi| |zenodo| |license|
===============================================================

.. |ci| image:: https://github.com/radionets-project/pyvisgrid/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/radionets-project/pyvisgrid/actions/workflows/ci.yml?branch=main
    :alt: Test Status

.. |codecov| image:: https://codecov.io/github/radionets-project/pyvisgrid/badge.svg
    :target: https://codecov.io/github/radionets-project/pyvisgrid
    :alt: Code coverage

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/radionets-project/pyvisgrid/main.svg
    :target: https://results.pre-commit.ci/latest/github/radionets-project/pyvisgrid/main
    :alt: pre-commit.ci status

.. |pypi| image:: https://badge.fury.io/py/pyvisgrid.svg
   :target: https://badge.fury.io/py/pyvisgrid
   :alt: PyPI version

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17201290.svg
   :target: https://doi.org/10.5281/zenodo.17201290
   :alt: Zenodo DOI

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/license/mit
    :alt: License: MIT

Installation
============

You can install the necessary packages in a mamba/conda environment of your choice by executing

.. code::

  $ pip install -e .

We recommend using a conda/mamba environment with ``python`` version ``<=3.11``.

If you want to use features from the NRAO CASAtools package, make sure you are using ``python`` 3.10 or 3.11.

Example
========

The following images show the different images resulting from the gridding process
of an observation of the protoplanetary disk ``Elias 24`` of the ALMA Observatory as part of the `DSHARP <https://almascience.eso.org/almadata/lp/DSHARP/>`_.

Ungridded :math:`(u,v)` coverage of the observation
-------------------------------------------------------------

.. image:: ./assets/examples/elias24_example_ungridded_uv.png
	:width: 49.5%

Amplitude (left) and Phase (right) of the gridded visibilities
--------------------------------------------------------------

.. |mask_abs| image:: ./assets/examples/elias24_example_mask_abs.png
	:width: 49.5%

.. |mask_phase| image:: ./assets/examples/elias24_example_mask_phase.png
	:width: 49.5%


|mask_abs| |mask_phase|

Dirty image created from the :math:`(u,v)` coverage.
--------------------------------------------------------------

.. image:: ./assets/examples/elias24_example_dirty_image.png
	:width: 49.5%

Cleaned image
-------------------------------------------

.. image:: ./assets/examples/elias24_example_clean.png
	:width: 49.5%

Acknowledgment
===============
This research made use of the data provided by the DSHARP:
Andrews, S. M. et. al, “The Disk Substructures at High Angular Resolution Project (DSHARP). I. Motivation, Sample, Calibration, and Overview”, *The Astrophysical Journal*, vol. 869, no. 2, Art. no. L41, IOP, 2018. doi:`10.3847/2041-8213/aaf741 <https://doi.org/10.3847/2041-8213/aaf741>`_.

