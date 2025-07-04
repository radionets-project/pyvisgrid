===============================================
pyvisgrid |ci| |pre-commit| |codecov| |license|
===============================================

.. |ci| image:: https://github.com/radionets-project/pyvisgrid/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/radionets-project/pyvisgrid/actions/workflows/ci.yml?branch=main
    :alt: Test Status

.. |codecov| image:: https://codecov.io/github/radionets-project/pyvisgrid/badge.svg
    :target: https://codecov.io/github/radionets-project/pyvisgrid
    :alt: Code coverage

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/radionets-project/pyvisgrid/main.svg
    :target: https://results.pre-commit.ci/latest/github/radionets-project/pyvisgrid/main
    :alt: pre-commit.ci status

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/license/mit
    :alt: License: MIT

Installation
============

You can install the necessary packages in a conda environment of your choice by executing

.. code::

  $ pip install -e .

We recommend using a conda/mamba environment with ``python`` version ``<=3.11``.

If you want to use features from the NRAO CASAtools package, make sure you are using python 3.10 or 3.11.

Examples
========

The following images show the different images resulting from the gridding process
of a simulated observation of the protoplanetary disk **Elias 24** from the
`DSHARP <https://almascience.eso.org/almadata/lp/DSHARP/>`_.

The visibilities were generated using `pyvisgen <https://github.com/radionets-project/pyvisgen`_ and
gridded using ``pyvisgrid``.
