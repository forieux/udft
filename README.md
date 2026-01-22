# UDFT: Unitary Discrete Fourier Transform (and related)

![licence](https://img.shields.io/github/license/forieux/udft) ![pypi](https://img.shields.io/pypi/v/udft) ![status](https://img.shields.io/pypi/status/udft) ![version](https://img.shields.io/pypi/pyversions/udft) ![maintained](https://img.shields.io/maintenance/yes/2025) [![Documentation Status](https://readthedocs.org/projects/udft/badge/?version=latest)](https://udft.readthedocs.io/en/latest/?badge=latest)

This module implements unitary discrete Fourier transform, that is orthonormal
`det(F) = 1` and `F⁻¹ = F^*`. This module existed before the introduction of the
`norm="ortho"` keyword and is now a very (very) thin wrapper around libraries.

It is useful for convolution [1]: they respect the Perceval equality, e.g., the
value of the null frequency is equal to `1/√N * ∑ₙ xₙ`.

```
[1] B. R. Hunt "A matrix theory proof of the discrete convolution theorem", IEEE
Trans. on Audio and Electroacoustics, vol. au-19, no. 4, pp. 285-288, dec. 1971
```

There is also functions related to Fourier and convolution like `ir2fr`.

UDFT use [array API standard](https://data-apis.org/array-api/latest/). Thanks
to this last point, any array library that follow the standard (`Numpy`,
`PyTorch`, `cupy`, ...) can be used by `udft` that use their respective
namespace.

If you are having issues, please let me know

francois.orieux AT universite-paris-saclay.fr

## Installation and documentation

UDFT is just the file `udft.py` and depends on
[array-api-compat](https://data-apis.org/array-api-compat/) and numpy only and
therefor is compatible with `numpy`, `pytorch`, `jax`, `cupy`...

The API is simple and opinionated for good reason. If you need more parameters
or options, I simply encourage you to directly use API of your array library.

Documentation is [here](https://udft.readthedocs.io/en/stable/index.html). I
recommend using [poetry](https://python-poetry.org/) or
[uv](https://docs.astral.sh/uv/) for installation

```
   poetry add udft
```
or
```
   poetry add udft[scipy]
```
For a quick and dirty installation, just copy the `udft.py` file: it is quite
stable, follow the [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
and futur major changes are unlikely.

The code is hosted on [GitHub](https://github.com/forieux/udft).

## License

The code is in the public domain.
