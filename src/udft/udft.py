# UDFT: Unitary Discrete Fourier Transform (and related)
# Copyright (C) 2021-2025 François Orieux <francois.orieux@universite-paris-saclay.fr>

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
-- ∑ₙ xₙ.
√N

The transforms are applied on the last axes for performances (C-order array).

This module use the Array API standard and use the array-api-compat module to be
array agnostic. For more flexible usage, you must use methods provided by your
array library.

"""

from typing import Optional
from types import ModuleType
from typing import TypeVar, Protocol, TypeGuard, Any
from collections.abc import Sequence

import array_api_compat
from array_api_compat import numpy as np

try:
    from scipy import fft as spfft
except ImportError:
    spfft: Optional[ModuleType] = None


__all__ = [  # noqa: WPS410
    "dftn",
    "idftn",
    "dft",
    "idft",
    "dft2",
    "idft2",
    "rdftn",
    "irdftn",
    "rdft",
    "rdft2",
    "hnorm",
    "crandn",
    "ir2fr",
    "fr2ir",
    "diff_ir",
    "laplacian",
]


class ArrayLike(Protocol):
    ndim: int
    shape: tuple[int, ...]

    def __getitem__(self, index: Any) -> Any: ...


Array = TypeVar("Array", bound=ArrayLike)


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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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

    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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

    return xp.fft.irfftn(
        inarray, s=shape, axes=range(-ndim, 0), norm="ortho", workers=-1
    )


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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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
    If `inarray` is a Numpy array, if available the multithreaded `scipy.fft` is
    used, otherwise the namespace of the array's library is used.

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

      3. Perform the DFT on the last axes

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
        origin = [ndim // 2 for ndim in imp_resp.shape[-ndim:]]

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
    irpadded = xp.zeros(imp_resp.shape[:-ndim] + shape)  # zeros of target shape
    irpadded[tuple(slice(0, axe) for axe in imp_resp.shape)] = imp_resp

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
    for axe, shift in enumerate(origin):
        irpadded = xp.roll(irpadded, -shift, imp_resp.ndim - ndim + axe)

    # Perform the DFT on the last axes
    if real:
        tf = xp.fft.rfftn(
            irpadded, axes=list(range(imp_resp.ndim - ndim, imp_resp.ndim))
        )
        return np.ascontiguousarray(tf, like=imp_resp)  # type: ignore
    tf = xp.fft.fftn(irpadded, axes=list(range(imp_resp.ndim - ndim, imp_resp.ndim)))
    return np.ascontiguousarray(tf, like=imp_resp)  # type: ignore


def fr2ir(
    freq_resp: Array,
    shape: tuple[int, ...],
    origin: Sequence[int] | None = None,
    real: bool = True,
) -> Array:
    """Return the impulse responses from frequency responses.

    This function makes the necessary correct zero-padding, zero convention,
    correct DFT etc. to compute the impulse responses from frequency responses.

    The IR array is supposed to have the origin in the middle of the array.

    The Fourier transform is performed on the last `len(shape)` dimensions for
    efficiency (C-order array). Use `freq_resp` namespacsp``` namespace and
    return an array like `freq_resp`.

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
        origin = [int(xp.floor(length / 2)) for length in shape]

    if len(origin) != ndim:
        raise ValueError("`origin` and `shape` must have the same length")

    if real:
        irpadded = xp.fft.irfftn(
            freq_resp, axes=list(range(freq_resp.ndim - ndim, freq_resp.ndim))
        )
    else:
        irpadded = xp.fft.ifftn(
            freq_resp, axes=list(range(freq_resp.ndim - ndim, freq_resp.ndim))
        )

    for axe, shift in enumerate(origin):
        irpadded = xp.roll(irpadded, shift, freq_resp.ndim - ndim + axe)

    return np.ascontiguousarray(
        a=irpadded[tuple(slice(0, length) for length in shape)],
        like=freq_resp,
    )  # ty:ignore[no-matching-overload]


def diff_ir(ndim=1, axis=0, norm=True, like=None):
    """Return the impulse response of first order differences.

    Parameters
    ----------
    ndim : int, optional
        The number of dimensions of the array on which the diff will apply.
    axis : int, optional
        The axis (dimension) where the diff operates.
    norm: bool, optional
        The output is normalized by ∑_i |h_i|.

    Returns
    -------
    out : array_like
        The impulse response
    """
    xp = array_api_compat.array_namespace(like) if like is not None else np

    if ndim <= 0:
        raise ValueError("The number of dimensions `ndim` must respect `ndim > 0`.")
    if axis >= ndim:
        raise ValueError("The `axis` argument must respect `0 <= axis < ndim`.")

    shape = ndim * [1]
    shape[axis] = 3
    if norm:
        return xp.reshape(xp.array([0, -1, 1], ndmin=ndim) / 2, shape)

    return xp.reshape(xp.array([0, -1, 1], ndmin=ndim), shape)


def laplacian(ndim: int, norm=False, like: Array | None = None) -> Array:
    """Return the Laplacian impulse response.

    The second-order difference in each axes.

    Parameters
    ----------
    ndim : int
        The dimension of the Laplacian.
    norm: bool, optional
        The output is normalized by ∑_i |h_i|.

    Returns
    -------
    out : array_like
        The impulse response
    """
    xp = array_api_compat.array_namespace(like) if like is not None else np

    imp = xp.zeros([3 for _ in range(ndim)])
    for dim in range(ndim):
        idx = tuple(
            [slice(1, 2) for _ in range(ndim)]
            + [slice(None)]
            + [slice(1, 2) for _ in range(ndim - dim - 1)]
        )
        imp[idx] = xp.array([-1.0, 0, -1.0]).reshape(
            [-1 if axe == dim else 1 for axe in range(ndim)]
        )
    imp[tuple([slice(1, 2) for _ in range(ndim)])] = 2.0 * ndim
    if norm:
        return imp / xp.sum(xp.abs(imp))

    return imp


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

    xp = array_api_compat.array_namespace()

    axis = tuple(range(-len(inshape), 0))
    axis2 = tuple(range(-(len(inshape) - 1), 0))

    norm = 2 * xp.sum(xp.abs(inarray) ** 2, axis=axis)
    norm -= xp.sum(xp.abs(inarray[..., 0]) ** 2, axis=axis2)

    if inshape[-1] % 2 == 0:
        norm -= xp.sum(xp.abs(inarray[..., -1]) ** 2, axis=axis2, keepdims=True)

    return xp.sqrt(norm)


def crandn(shape: tuple[int, ...], like: Array | None = None) -> Array:
    """Draw from white complex Normal.

    Draw unitary DFT of real white Gaussian field of zero mean and unit
    variance. Does not consider hermitian property, `shape` is supposed to
    consider half of the frequency plane already.
    """
    xp = array_api_compat.array_namespace(like) if like else np

    return xp.sqrt(0.5) * (
        xp.random.standard_normal(shape) + 1j * xp.random.standard_normal(shape)
    )


### Local Variables:
### ispell-local-dictionary: "english"
### End:
