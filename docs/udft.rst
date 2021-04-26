.. py:currentmodule:: udft

The ``LIB`` module variable
---------------------------

The ``LIB`` variable specifies the default library to use by the module. The Numpy
library is used by default with ``LIB`` sets to ``"numpy"``. Possible values are

- ``"numpy"`` to use Numpy, or
- ``"fftw"`` to use `pyFFTW <https://pypi.org/project/pyFFTW/>`_, if installed.

The variable can be changed globally

.. code-block:: python

   udft.LIB = "numpy"

In addition, each function has a parameter to change the library used at call
time. A ``ValueError`` is raided if the value is not recognized.

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
transformed axis has length ``N // 2 + 1``. For inverse transform, the shape
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
