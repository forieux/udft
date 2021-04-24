.. py:currentmodule:: udft

The ``LIB`` module variable
---------------------------

The ``LIB`` variable is a string set at import time. If `pyFFTW
<https://pypi.org/project/pyFFTW/>`_ is installed, this library is used by
default and ``LIB`` has the value ``"ffwt"``. Otherwise ``numpy`` is used and
``LIB`` takes the value ``"numpy"``.

The variable can be change globally

.. code-block:: python

   udft.LIB = "numpy"

In addition, each function has a parameter to change the library used at call
time. The accepted values are ``"fftw"`` and ``"numpy"``, a ``ValueError`` is
raided otherwise.

Discrete Fourier Transform
--------------------------

.. autofunction:: dftn

.. autofunction:: idftn

.. autofunction::  dft

.. autofunction::  idft

.. autofunction::  dft2

.. autofunction::  idft2

Real Discrete Fourier Transform
-------------------------------

The transform here suppose input of real values. In direct transform, the last
transformed axis has lenght ``N // 2 + 1``. For inverse transform, the shape
must be provided.

.. autofunction:: rdftn

.. autofunction:: irdftn

.. autofunction::  rdft

.. autofunction::  rdft2

Convolution related
-------------------

.. autofunction::  ir2fr

.. autofunction::  fr2ir

.. autofunction::  diff_ir

.. autofunction::  laplacian

Other
-----

.. autofunction::  norm

.. autofunction::  crandn
