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

dftn
~~~~

.. autofunction:: dftn

idftn
~~~~~

.. autofunction:: idftn

dft
~~~

.. autofunction::  dft

idft
~~~~

.. autofunction::  idft

dft2
~~~~

.. autofunction::  dft2

idft2
~~~~~

.. autofunction::  idft2

Real Discrete Fourier Transform
-------------------------------

The transforms here suppose input of real values. In direct transform, the last
transformed axis has length ``N // 2 + 1``.

.. note::

   The exact output shape for real transform can't be guessed from input shape.
   Therefor, only one inverse transform :func:`irdftn` is provided and that
   function ask for the output shape. The dimension `ndim` corresponds to the
   length of shape.

rdftn
~~~~~

.. autofunction:: rdftn

irdftn
~~~~~~

.. autofunction:: irdftn

rdft
~~~~

.. autofunction::  rdft

rdft2
~~~~~

.. autofunction::  rdft2

Convolution related
-------------------

ir2fr
~~~~~

.. autofunction::  ir2fr

fr2ir
~~~~~

.. autofunction::  fr2ir

diff_ir
~~~~~~~

.. autofunction::  diff_ir

laplacian
~~~~~~~~~

.. autofunction::  laplacian

Other
-----

hnorm
~~~~~

.. autofunction::  hnorm

crandn
~~~~~~

.. warning::

   At that time, this function does not respect the hermitian property, this is
   planned.

.. autofunction::  crandn
