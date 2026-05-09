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

.. |maintained| image:: https://img.shields.io/maintenance/yes/2026
   :alt: Maintained

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :alt: Documentation Status
   :target: https://docs.readthedocs.io/en/latest/?badge=latest

This module implements unitary (orthonormal) discrete Fourier transforms and
related functions for convolution. It is built on top of the `Array API standard
<https://data-apis.org/array-api/latest/>`_ via `array-api-compat
<https://data-apis.org/array-api-compat/>`_, making it compatible with NumPy,
PyTorch, and other compliant array libraries.

It is useful for convolution [1]: they respect the Parseval equality
:math:`\|x\|_2^2 = \|X\|_2^2`, e.g., the value of the null frequency :math:`X_0`
is equal to

.. math::

   X_0 = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} x_n, \text{ and } x_0 = \frac{1}{\sqrt{N}} \sum_{n'=0}^{N-1} X_{n'}.

and if :math:`H` is a circulant convolution with :math:`h` as a real impulse
response, then :math:`H = F^* \Lambda F` where :math:`F^*` is the unitary IDFT
computed by :func:`irdftn`, :math:`F` the unitary DFT computed by :func:`rdftn`,
and :math:`\Lambda` the frequency response computed with :func:`ir2fr` from
:math:`h`.

::

   [1] B. R. Hunt "A matrix theory proof of the discrete convolution theorem",
   IEEE Trans. on Audio and Electroacoustics, vol. au-19, no. 4, pp. 285-288,
   dec. 1971

If you are having issues, please let me know:

francois.orieux AT universite-paris-saclay.fr

Installation
============

UDFT is a single file (``udft.py``) requiring Python >= 3.12. Install with pip:

.. code-block:: sh

   pip install udft

For multithreaded FFT on NumPy arrays, install the optional SciPy dependency:

.. code-block:: sh

   pip install udft[scipy]

The project follows `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

License
=======

The code is in the public domain.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   udft
