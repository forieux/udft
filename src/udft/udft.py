# UDFT: Unitary Discrete Fourier Transform (and related)
# Copyright (C) 2021-2026 François Orieux <francois.orieux@universite-paris-saclay.fr>

# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <https://unlicense.org>

"""UDFT
====

Unitary discrete Fourier transform (and related)

This module implements unitary discrete Fourier transforms, which are
orthonormal. They are useful for convolution: they respect the Parseval
equality, the value of the null frequency is equal to

 1
-- ∑ₙ xₙ
√N

and if H is a circulant convolution with h as a real impulse response, then H =
F^* Λ F and F^* is the unitary IDFT computed by `irdftn`, F the unitary DFT
computed by `rdftn` and Λ the frequency response computed with `ir2fr` from h.

The transforms are applied on the last axes for performances (C-order array).

This module use the Array API standard and use the array-api-compat module to be
array agnostic. For more flexible usage, you must use methods provided by your
array library.

"""

import math
from collections.abc import Sequence
from types import ModuleType
from typing import Any, TypeGuard

import array_api_compat
from array_api_compat import numpy as np

spfft: ModuleType | None

try:
    from scipy import fft as spfft
except ImportError:
    spfft = None


__all__ = [  # noqa: WPS410
    "crandn",
    "dft",
    "dft2",
    "dftn",
    "diff_ir",
    "fr2ir",
    "hnorm",
    "idft",
    "idft2",
    "idftn",
    "ir2fr",
    "irdftn",
    "laplacian",
    "rdft",
    "rdft2",
    "rdftn",
]


type Array = Any  # array API standard array — no stable cross-backend type yet


def is_array(arr: Array) -> TypeGuard[Array]:
    """A TypeGuard for array-like objects."""
    return array_api_compat.is_array_api_obj(arr)


_not_array_message = (
    "`inarray` must be a compatible with Array API Standard (eg. numpy, pytorch, ...)"
)


def dftn(
    inarray: Array,
    ndim: int | None = None,
) -> Array:
    """ND unitary discrete Fourier transform.

    Parameters
    ----------
    inarray : array-like
        The array to transform.
    ndim : int, optional
        The `ndim` last axes along which to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like
        The DFT of `inarray` with same shape.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    if not is_array(inarray):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(inarray)

    ndim = inarray.ndim if ndim is None else ndim

    if ndim < 1 or ndim > inarray.ndim:
        raise ValueError("`ndim` must be >= 1.")

    if array_api_compat.is_numpy_array(inarray) and spfft:
        return spfft.fftn(inarray, axes=range(-ndim, 0), norm="ortho", workers=-1)

    return xp.fft.fftn(inarray, axes=range(-ndim, 0), norm="ortho")


def idftn(
    inarray: Array,
    ndim: int | None = None,
) -> Array:
    """ND unitary inverse discrete Fourier transform.

    Parameters
    ----------
    inarray : array-like
        The array to transform.
    ndim : int, optional
        The `ndim` last axes along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like
        The IDFT of `inarray` with same shape.

    Notes
    -----

    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    if not is_array(inarray):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(inarray)

    ndim = inarray.ndim if ndim is None else ndim

    if ndim < 1 or ndim > inarray.ndim:
        raise ValueError("`ndim` must be >= 1.")

    if array_api_compat.is_numpy_array(inarray) and spfft:
        return spfft.ifftn(inarray, axes=range(-ndim, 0), norm="ortho", workers=-1)

    return xp.fft.ifftn(inarray, axes=range(-ndim, 0), norm="ortho")


def dft(inarray: Array) -> Array:
    """1D unitary discrete Fourier transform.

    Compute the unitary DFT on the last axis.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The DFT of `inarray` with same shape.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return dftn(inarray, 1)


def idft(inarray: Array) -> Array:
    """1D unitary inverse discrete Fourier transform.

    Compute the unitary inverse DFT transform on the last axis.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The IDFT of `inarray` with same shape.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return idftn(inarray, 1)


def dft2(inarray: Array) -> Array:
    """2D unitary discrete Fourier transform.

    Compute the unitary DFT on the last 2 axes.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The DFT of `inarray` with same shape.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return dftn(inarray, 2)


def idft2(inarray: Array) -> Array:
    """2D unitary inverse discrete Fourier transform.

    Compute the unitary IDFT on the last 2 axes.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The IDFT of `inarray` with same shape.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return idftn(inarray, 2)


def rdftn(
    inarray: Array,
    ndim: int | None = None,
) -> Array:
    """ND real unitary discrete Fourier transform.

    Consider the Hermitian property of output with input having real values.

    Parameters
    ----------
    inarray : array-like
        The array of real values to transform.
    ndim : int, optional
        The `ndim` last axes along which to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like
        The real DFT of `inarray` (the last axe as N // 2 + 1 length).

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    if not is_array(inarray):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(inarray)

    ndim = inarray.ndim if ndim is None else ndim

    if ndim < 1 or ndim > inarray.ndim:
        raise ValueError("`ndim` must be >= 1.")

    if array_api_compat.is_numpy_array(inarray) and spfft:
        return spfft.rfftn(inarray, axes=range(-ndim, 0), norm="ortho", workers=-1)

    return xp.fft.rfftn(inarray, axes=range(-ndim, 0), norm="ortho")


def irdftn(
    inarray: Array,
    shape: tuple[int, ...],
) -> Array:
    """ND real unitary inverse discrete Fourier transform.

    Consider the Hermitian property of input and return real values.

    Parameters
    ----------
    inarray : array-like
        The array of complex values to transform.
    shape : tuple of int
        The output shape of the `len(shape)` last axes. The transform is applied
        on the `n=len(shape)` axes.

    Returns
    -------
    outarray : array-like
        The real IDFT of `inarray`.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    if not is_array(inarray):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(inarray)
    ndim = len(shape)

    if ndim < 1 or inarray.ndim < ndim:
        raise ValueError("`shape` must respect `1 <= ndim <= inarray.ndim`.")

    if array_api_compat.is_numpy_array(inarray) and spfft:
        return spfft.irfftn(
            inarray,
            s=shape,
            axes=range(-ndim, 0),
            norm="ortho",
            workers=-1,
        )

    return xp.fft.irfftn(inarray, s=shape, axes=range(-ndim, 0), norm="ortho")


def rdft(inarray: Array) -> Array:
    """1D real unitary discrete Fourier transform.

    Compute the unitary real DFT on the last axis. Consider the Hermitian
    property of output with input having real values.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The real DFT of `inarray`, where the last dim has length N//2+1.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return rdftn(inarray, 1)


def rdft2(inarray: Array) -> Array:
    """2D real unitary discrete Fourier transform.

    Compute the unitary real DFT on the last 2 axes. Consider the Hermitian
    property of output when input has real values.

    Parameters
    ----------
    inarray : array-like
        The array to transform.

    Returns
    -------
    outarray : array-like
        The real DFT of `inarray`, where the last dim has length N//2+1.

    Notes
    -----
    For NumPy arrays, `scipy.fft` is used when available (enabling multithreading);
    for other array types, the FFT is dispatched through the array library's namespace.

    """
    return rdftn(inarray, 2)


def ir2fr(
    imp_resp: Array,
    shape: tuple[int, ...],
    origin: Sequence[int] | None = None,
    real: bool = True,
) -> Array:
    """Compute the frequency response from impulse responses.

    This function makes the necessary correct zero-padding, zero convention,
    correct DFT etc.

    The DFT is performed on the last `len(shape)` dimensions for efficiency
    (C-order array). Use the `imp_resp` namespace and return a `imp_resp`
    array-like.

    If H is a circulant convolution with h as impulse response, then H = F^* Λ F
    and this function compute Λ from h when F and F^* are unitary DFT and IDFT.

    Parameters
    ----------
    imp_resp : array-like
        The impulse responses.
    shape : tuple of int
        A tuple of integer corresponding to the target shape of the frequency
        responses, without hermitian property. The DFT is performed on the
        `len(shape)` last axes of ndarray.
    origin : tuple of int, optional
        The index of the origin (0 coordinate) of the impulse response. The
        center of the array by default (`shape[i] // 2`).
    real : boolean, optional
        If True, `imp_resp` is supposed real, and real DFT is used.

    Returns
    -------
    out : array-like
      The frequency responses of shape `shape` on the last `len(shape)`
      dimensions. If `real` is `True`, the last dimension as length `N//2+1`.

    Notes
    -----
    - The output is returned as C-contiguous array.
    - For convolution, the result must be used with unitary discrete Fourier
      transform for the signal (`dftn` or equivalent).
    - What it does is

      1. Place the IR with zero filling on the target shape

         .. code-block:: text

            ┌────────┬──────────────┐
            │        │              │
            │   IR   │              │
            │        │              │
            │        │              │
            ├────────┘              │
            │            0          │
            │                       │
            │                       │
            │                       │
            └───────────────────────┘

      2. Roll (circshift in Matlab) to move the origin at index 0 (DFT hypothesis)

         .. code-block:: text

            ┌────────┬──────────────┐     ┌────┬─────────────┬────┐
            │11112222│              │     │4444│             │3333│
            │11112222│              │     │4444│             │3333│
            │33334444│              │     ├────┘             └────┤
            │33334444│              │     │                       │
            ├────────┘   0          │ ->  │           0           │
            │                       │     │                       │
            │                       │     ├────┐             ┌────┤
            │                       │     │2222│             │1111│
            │                       │     │2222│             │1111│
            └───────────────────────┘     └────┴─────────────┴────┘

      3. Perform the DFT with 1/N normalisation on the last axes

      4. Return the result as a contiguous array

    """
    if not is_array(imp_resp):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(imp_resp)
    ndim = len(shape)

    if ndim > imp_resp.ndim:
        raise ValueError(
            f"length ({ndim}) of `shape` must be inferior or equal to `imp_resp.ndim` ({imp_resp.ndim})"
        )

    if origin is None:
        origin = [length // 2 for length in imp_resp.shape[-ndim:]]

    if len(origin) != ndim:
        raise ValueError("`origin` and `shape` must have the same length")

    # Place the IR at the beginning of irpadded
    # ┌────────┬──────────────┐
    # │        │              │
    # │   IR   │              │
    # │        │              │
    # │        │              │
    # ├────────┘              │
    # │            0          │
    # │                       │
    # │                       │
    # │                       │
    # └───────────────────────┘
    # Same dtype and device as `imp_resp` (default float dtype for integer input)
    dtype = (
        imp_resp.dtype
        if xp.isdtype(imp_resp.dtype, ("real floating", "complex floating"))
        else None
    )
    irpadded = xp.zeros(
        tuple(imp_resp.shape[:-ndim]) + tuple(shape),
        dtype=dtype,
        device=array_api_compat.device(imp_resp),
    )
    idx = tuple(slice(0, axe) for axe in imp_resp.shape)
    if array_api_compat.is_jax_array(irpadded):
        irpadded = irpadded.at[idx].set(imp_resp)  # JAX arrays are immutable
    else:
        irpadded[idx] = imp_resp

    # Roll (circshift in Matlab) to move the origin at index 0 (DFT hypothesis)
    # ┌────────┬──────────────┐     ┌────┬─────────────┬────┐
    # │11112222│              │     │4444│             │3333│
    # │11112222│              │     │4444│             │3333│
    # │33334444│              │     ├────┘             └────┤
    # │33334444│              │     │                       │
    # ├────────┘   0          │ ->  │           0           │
    # │                       │     │                       │
    # │                       │     ├────┐             ┌────┤
    # │                       │     │2222│             │1111│
    # │                       │     │2222│             │1111│
    # └───────────────────────┘     └────┴─────────────┴────┘
    axes = tuple(range(-ndim, 0))
    irpadded = xp.roll(irpadded, shift=tuple(-shift for shift in origin), axis=axes)

    # Perform the DFT on the last axes
    if real:
        return xp.fft.rfftn(irpadded, axes=axes, norm="backward")
    return xp.fft.fftn(irpadded, axes=axes, norm="backward")


def fr2ir(
    freq_resp: Array,
    shape: tuple[int, ...],
    origin: Sequence[int] | None = None,
    real: bool = True,
    inshape: tuple[int, ...] | None = None,
) -> Array:
    """Return the impulse responses from frequency responses.

    This function makes the necessary correct zero-padding, zero convention,
    correct DFT etc. to compute the impulse responses from frequency responses.

    The IR array is supposed to have the origin in the middle of the array.

    The Fourier transform is performed on the last `len(shape)` dimensions for
    efficiency (C-order array). Use `freq_resp` namespace and return an array
    like `freq_resp`.

    Parameters
    ----------
    freq_resp : array-like
       The frequency responses.
    shape : tuple of int
       Output shape of the impulse responses.
    origin : tuple of int, optional
        The index of the origin (0, 0) of output the impulse response. The center by
        default (shape[i] // 2).
    real : boolean, optional
       If True, imp_resp is supposed real, and real DFT is used.
    inshape : tuple of int, optional
       The shape of the full frequency grid on the last `len(shape)` axes, that
       is the `shape` given to `ir2fr`. Only used if `real` is True, where the
       length of the last axis can't be guessed from `freq_resp`: it is
       `2 * (M - 1)` by default, with `M` the length of the last axis of
       `freq_resp`, and is therefore required if that length is odd.

    Returns
    -------
    out : array-like
       The impulse responses of shape `shape` on the last `len(shape)` axes.

    Notes
    -----
    - The output is returned as C-contiguous array.
    - For convolution, the result has to be used with unitary discrete Fourier
      transform for the signal (udftn or equivalent).

    """
    if not is_array(freq_resp):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(freq_resp)

    ndim = len(shape)

    if ndim > freq_resp.ndim:
        raise ValueError(
            "length of `shape` must be inferior or equal to `freq_resp.ndim`"
        )

    if origin is None:
        origin = [length // 2 for length in shape]

    if len(origin) != ndim:
        raise ValueError("`origin` and `shape` must have the same length")

    if inshape is not None and len(inshape) != ndim:
        raise ValueError("`inshape` and `shape` must have the same length")

    axes = tuple(range(-ndim, 0))

    if real:
        irpadded = xp.fft.irfftn(freq_resp, s=inshape, axes=axes, norm="backward")
    else:
        irpadded = xp.fft.ifftn(freq_resp, axes=axes, norm="backward")

    irpadded = xp.roll(irpadded, shift=tuple(origin), axis=axes)

    # Copy so the output is not a view of the padded array (C-contiguous in practice)
    return xp.asarray(
        irpadded[(..., *(slice(0, length) for length in shape))], copy=True
    )


def _asarray_like(arr: Array, like: Array | None) -> Array:
    """Return NumPy `arr` with namespace and device of `like`.

    The dtype of `like` is used if real floating. Return `arr` if `like` is None.
    """
    if like is None:
        return arr

    xp = array_api_compat.array_namespace(like)
    dtype = like.dtype if xp.isdtype(like.dtype, "real floating") else None
    return xp.asarray(arr, dtype=dtype, device=array_api_compat.device(like))


def diff_ir(
    ndim: int = 1, axis: int = 0, norm: bool = True, like: Array | None = None
) -> Array:
    """Return the impulse response of first order differences.

    Parameters
    ----------
    ndim : int, optional
        The number of dimensions of the array on which the diff will apply.
    axis : int, optional
        The axis (dimension) where the diff operates.
    norm: bool, optional
        The output is normalized by :math:`\\sum_i |h_i|`.
    like : array-like, optional
        Return an array with the namespace and device of `like`, and its dtype
        if real floating. NumPy by default.

    Returns
    -------
    out : array_like
        The impulse response
    """
    if ndim <= 0:
        raise ValueError("The number of dimensions `ndim` must respect `ndim > 0`.")
    if axis >= ndim:
        raise ValueError("The `axis` argument must respect `0 <= axis < ndim`.")

    shape = ndim * [1]
    shape[axis] = 3
    imp = np.reshape(np.asarray([0.0, -1.0, 1.0]), tuple(shape))
    if norm:
        imp = imp / 2

    return _asarray_like(imp, like)


def laplacian(ndim: int, norm=False, like: Array | None = None) -> Array:
    """Return the Laplacian impulse response.

    The second-order difference in each axes.

    Parameters
    ----------
    ndim : int
        The dimension of the Laplacian.
    norm: bool, optional
        The output is normalized by :math:`\\sum_i |h_i|`.
    like : array-like, optional
        Return an array with the namespace and device of `like`, and its dtype
        if real floating. NumPy by default.

    Returns
    -------
    out : array_like
        The impulse response
    """
    imp = np.zeros((3,) * ndim)
    center = (1,) * ndim
    for dim in range(ndim):
        for side in (0, 2):
            imp[center[:dim] + (side,) + center[dim + 1 :]] = -1.0
    imp[center] = 2.0 * ndim
    if norm:
        imp = imp / np.sum(np.abs(imp))

    return _asarray_like(imp, like)


def hnorm(inarray: Array, inshape: tuple[int, ...]) -> Array:
    r"""Hermitian l2-norm of array in discrete Fourier space.

    Compute the l2-norm of complex array

    .. math::

       \|x\|_2 = \sqrt{\sum_{n=1}^{N} |x_n|^2}

    considering the Hermitian property. Must be used with `rdftn`. Equivalent of
    `np.linalg.norm` for array applied on full Fourier space array (those
    obtained with `dftn`).

    Parameters
    ----------
    inarray : array-like of shape (... + inshape)
        The input array with half of the Fourier plan.

    inshape: tuple of int
        The shape of the original array `oarr` where `inarray=rdft(oarr)`.

    Returns
    -------
    norm : float
    """
    if not is_array(inarray):
        raise ValueError(_not_array_message)

    xp = array_api_compat.array_namespace(inarray)

    axis = tuple(range(-len(inshape), 0))
    axis2 = tuple(range(-(len(inshape) - 1), 0))

    norm = 2 * xp.sum(xp.abs(inarray) ** 2, axis=axis)
    norm = norm - xp.sum(xp.abs(inarray[..., 0]) ** 2, axis=axis2)

    if inshape[-1] % 2 == 0:
        norm = norm - xp.sum(xp.abs(inarray[..., -1]) ** 2, axis=axis2)

    return xp.sqrt(norm)


def crandn(shape: tuple[int, ...], like: Array | None = None) -> Array:
    """Draw from white complex Normal.

    Draw unitary DFT of real white Gaussian field of zero mean and unit
    variance. Does not consider hermitian property, `shape` is implicity
    supposed to be half of the frequency plane already or the full plane if the
    hermitian property is not present.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output.
    like : array-like, optional
        Return an array with the namespace and device of `like`, complex64 if
        `like` is float32 or complex64, complex128 otherwise. NumPy by default.
    """
    if like is None:
        return math.sqrt(0.5) * (
            np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)
        )

    xp = array_api_compat.array_namespace(like)
    device = array_api_compat.device(like)
    # complex64 for float32 or complex64 `like`, complex128 otherwise
    if xp.isdtype(like.dtype, ("real floating", "complex floating")):
        dtype = xp.result_type(like.dtype, xp.complex64)
    else:
        dtype = xp.complex128

    if array_api_compat.is_torch_array(like):
        # For complex dtype, torch draws real and imaginary parts of variance 1/2
        return xp.randn(shape, dtype=dtype, device=device)

    # Draw on device with numpy-like random module (numpy, cupy), else with numpy
    # then transfer (eg. JAX that has no global random state)
    rnd = xp.random if hasattr(xp, "random") else np.random
    arr = math.sqrt(0.5) * (
        rnd.standard_normal(shape) + 1j * rnd.standard_normal(shape)
    )
    return xp.asarray(arr, dtype=dtype, device=device)


### Local Variables:
### ispell-local-dictionary: "english"
### End:
