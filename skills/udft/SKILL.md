---
name: udft
description: Write correct code with udft, the Python library of unitary (orthonormal, norm="ortho") discrete Fourier transforms and circulant-convolution helpers (dftn, idftn, rdftn, irdftn, ir2fr, fr2ir, laplacian, diff_ir, hnorm, crandn). Use when code imports udft, or when a task needs Parseval-preserving FFTs, FFT-based circular convolution or deconvolution (Wiener, Tikhonov, quadratic regularization), or conversion between impulse response (PSF, kernel) and frequency response (transfer function).
license: Unlicense
metadata:
  author: François Orieux
  repository: https://github.com/forieux/udft
---

# udft

`udft` is a single-module Python library (`pip install udft`, or `udft[scipy]`
for multithreaded FFTs on NumPy arrays; Python >= 3.12). Import it with
`import udft`: every public function is at the top level. The API is small and
opinionated on purpose: when an option is missing (arbitrary axes, padding,
another normalization), call the array library's `fft` module directly instead
of working around udft.

## Conventions to respect

1. **Unitary transforms.** `dftn`, `rdftn`, `idftn`, `irdftn` (and the 1D/2D
   shortcuts) use `norm="ortho"`. Parseval holds, `‖x‖₂ = ‖dftn(x)‖₂`, and the
   zero frequency is `sum(x) / sqrt(N)`. Compared to NumPy defaults,
   `udft.dftn(x) == np.fft.fftn(x) / sqrt(N)`, where `N` is the number of
   samples in the transformed axes. Never mix the two in one computation.
2. **Last axes only.** Transforms act on the `ndim` *trailing* axes (all axes
   by default). There is no `axis`/`axes` argument: leading axes are batch
   axes. `dft`/`idft`/`rdft` use the last axis, `dft2`/`idft2`/`rdft2` the
   last two.
3. **Real transforms keep half the spectrum.** `rdftn` returns a last axis of
   length `N // 2 + 1`. The only real inverse is `irdftn(X, shape)`, and
   `shape` is **required**: it gives the output size of the transformed
   trailing axes (`len(shape)` is the number of transformed axes).
4. **Array namespace follows the input.** NumPy in gives NumPy out, PyTorch in
   gives PyTorch out (through `array-api-compat`). `laplacian`, `diff_ir` and
   `crandn` create arrays and return NumPy unless `like=` is given; they rely on
   NumPy-specific functions, so treat them as NumPy-only. NumPy is the backend
   covered by the tests.

## Convolution model

For a circular convolution `y = H x`, `H = F* Λ F` where `F` is `rdftn` (or
`dftn`), `F*` is `irdftn` (or `idftn`) and `Λ = ir2fr(h, shape)`.

`ir2fr(imp_resp, shape, origin=None, real=True)`:

- zero-pads `imp_resp` to `shape` on its trailing axes, then rolls it so that
  `origin` lands at index 0, then takes the DFT **without normalization**
  (Σₙ, the MATLAB `fft` convention, NumPy `norm="backward"`);
- `origin` defaults to the center of the impulse response, `imp_resp.shape[i] // 2`
  per axis. For an even-sized kernel, pass `origin` explicitly;
- the result is `y[n] = Σᵢ h[i] · x[(n − (i − origin)) mod N]`, a convolution
  with the kernel centered on `origin`;
- `real=True` (default) returns the half spectrum matching `rdftn`; use
  `real=False` with `dftn`/`idftn` (complex kernel or full spectrum);
- `imp_resp` may have more axes than `len(shape)`: the extra leading axes are
  kept, giving a batch of frequency responses that broadcasts against data.

`fr2ir(freq_resp, shape, origin=None, real=True, inshape=None)` is the inverse:
it returns the impulse response of size `shape` with its origin at
`shape[i] // 2`. With `real=True`, pass `inshape` (the `shape` given to
`ir2fr`) whenever the last axis of the grid may be odd: a half spectrum alone
does not tell its original length.

## Recipes

```python
import numpy as np
import udft

x = np.random.standard_normal((256, 256))
h = np.ones((5, 5)) / 25                   # 5x5 kernel, origin at its center

# Circular convolution
fr = udft.ir2fr(h, x.shape)                # shape (256, 129)
y = udft.irdftn(fr * udft.rdftn(x), x.shape)

# Quadratic regularized deconvolution (Tikhonov / Wiener-like),
# with a Laplacian penalty and regularization weight mu
mu = 0.1
lap = udft.ir2fr(udft.laplacian(2), x.shape)
x_hat = udft.irdftn(
    np.conj(fr) * udft.rdftn(y) / (np.abs(fr) ** 2 + mu * np.abs(lap) ** 2),
    x.shape,
)

# Gradient-type penalties: first-order differences along each axis
d0 = udft.ir2fr(udft.diff_ir(ndim=2, axis=0), x.shape)
d1 = udft.ir2fr(udft.diff_ir(ndim=2, axis=1), x.shape)

# Norm in Fourier space (Hermitian symmetry handled)
assert np.isclose(udft.hnorm(udft.rdftn(x), x.shape), np.linalg.norm(x))

# Batch of images: transform the last two axes only
stack = np.random.standard_normal((10, 64, 64))
S = udft.dft2(stack)                       # same as udft.dftn(stack, ndim=2)
```

Because the transforms are unitary, a white Gaussian noise of variance σ² stays
white with variance σ² in Fourier space. `crandn(shape)` draws such a complex
normal sample directly (variance 1, no Hermitian symmetry enforced).

## Pitfalls

- Forgetting `shape` in `irdftn` (or `inshape` in `fr2ir` with `real=True`),
  or passing the half-spectrum shape instead of the original data shape.
- Passing an axis index as `ndim`: it is a *count* of trailing axes.
- Using `numpy.linalg.norm` on a half spectrum from `rdftn`: use `hnorm`.
  `numpy.linalg.norm` is correct only on full spectra from `dftn`.
- Multiplying `ir2fr` output by an extra `sqrt(N)` or `N`: with unitary `F`,
  `irdftn(ir2fr(h, s) * rdftn(x), s)` is already the convolution.
- `ir2fr` creates its padding with the namespace default float dtype; cast the
  result if the computation must stay in `float32`/`complex64`.

## Reference

- Full API with docstrings: the source file
  <https://raw.githubusercontent.com/forieux/udft/main/src/udft/udft.py>
- Rendered docs: <https://udft.readthedocs.io/en/stable/udft.html>
- Documentation index for agents: <https://udft.readthedocs.io/en/stable/llms.txt>
