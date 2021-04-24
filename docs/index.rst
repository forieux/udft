========================================================
 UDFT: Unitary Discrete Fourier Transform (and related)
========================================================

|licence| |pypi| |status| |version| |maintained| |docs|

.. |licence| image:: https://img.shields.io/github/license/forieux/udft
   :alt: Documentation Status

.. |pypi| image:: https://img.shields.io/pypi/v/udft
   :alt: Pypi version

.. |status| image:: https://img.shields.io/pypi/status/udft
   :alt: Status of the code

.. |version| image:: https://img.shields.io/pypi/pyversions/udft
   :alt: Version number

.. |maintained| image:: https://img.shields.io/maintenance/yes/2021
   :alt: Maintained

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :alt: Documentation Status
   :scale: 100%
   :target: https://docs.readthedocs.io/en/latest/?badge=latest

This module implements unitary discrete Fourier transform, that is orthonormal.
This module existed before the introduction of the ``norm="ortho"`` keyword and
is now a very (very) thin wrapper around Numpy or `pyFFTW
<https://pypi.org/project/pyFFTW/>`_ (maybe others in the future), mainly done
for my *personal usage*. There is also functions related to Fourier and
convolution like ``ir2fr``.

It is useful for convolution [1]: they respect the Perceval equality
:math:`\|x\|_2^2 = \|X\|_2^2`, e.g., the value of the null frequency :math:`X_0`
is equal to

.. math::

   X_0 = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} x_n, \text{ and } x_0 = \frac{1}{\sqrt{N}} \sum_{n'=0}^{N-1} X_{n'}.


::

   [1] B. R. Hunt "A matrix theory proof of the discrete convolution theorem",
   IEEE Trans. on Audio and Electroacoustics, vol. au-19, no. 4, pp. 285-288,
   dec. 1971

If you are having issues, please let me know

francois.orieux AT l2s.centralesupelec.fr

Installation and documentation
==============================

UDFT is just the file ``udft.py`` and depends on ``numpy`` and Python 3.7 only. I
recommend using poetry for installation

.. code-block:: sh

   poetry add udft

For potential better performance, `pyFFTW <https://pypi.org/project/pyFFTW/>`_
is optional and installable with

.. code-block:: sh

   poetry add udft[fftw]

The package is available with pip also. For a quick and dirty installation, just
copy the ``udft.py`` file: it is quite stable, follow the `Semantic Versioning
<https://semver.org/spec/v2.0.0.html>`_, and major changes are unlikely.

License
=======

The code is in the public domain.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   udft
