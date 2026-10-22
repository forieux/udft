# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [5.0.0] - 2026-10-22
### Fixed
- `dftn`, `idftn`, `rdftn`, `irdftn` (and their 1D/2D shortcuts) failed with
  PyTorch, which requires a tuple of axes, not a `range`.
- `ir2fr` creates its padding on the device and with the dtype of `imp_resp`:
  an impulse response on GPU gives a frequency response on the same GPU,
  float32 gives complex64 (complex128 before with NumPy), and a complex
  impulse response keeps its imaginary part with `onesided=False`.
- `ir2fr` and `fr2ir` no longer use `numpy.ascontiguousarray(like=)`, which
  only works with libraries implementing `__array_function__` (not PyTorch).
  `ir2fr` supports the immutable arrays of JAX.
- `hnorm` returns a scalar instead of a `(1,)` array when the last axis is
  even.
- `crandn(like=x)` failed for `x` with more than one element.
- `diff_ir`, `laplacian` and `crandn` with `like=` return an array on the
  device of `like`.
- `fr2ir` raises `ValueError` when `shape` exceeds the frequency grid.
- `laplacian` raises `ValueError` for `ndim <= 0`, like `diff_ir`.
- `diff_ir` accepts a negative `axis` (`-ndim <= axis < ndim`) and raises
  `ValueError` out of range, instead of an `IndexError`.
- `ir2fr`, `fr2ir` and `hnorm` raise a clear `ValueError` when `shape` (or
  `inshape`) is empty or longer than the number of axes of the input.
- Docstring of `ir2fr`.
- Clearer error message when `ndim` exceeds the number of axes in `dftn`,
  `idftn` and `rdftn`.

### Changed
- Require `array-api-compat>=1.13.0`: earlier versions do not recognize JAX
  arrays inside `jax.jit`.
- **Breaking:** the `real` keyword of `ir2fr` and `fr2ir` is renamed
  `onesided`. It selects the half plane (`rfftn`/`irfftn`, to use with
  `rdftn`/`irdftn`) or the full plane (`fftn`/`ifftn`), not the nature of
  the impulse response.
- `ir2fr` raises `ValueError` for a complex `imp_resp` with `onesided=True`
  (the imaginary part was silently discarded before).
- `diff_ir(norm=False)` returns floats instead of integers.
- `crandn(like=x)` with a PyTorch tensor draws with `torch.randn`.

## [4.1.0] - 2026-10-15
### Fixed
Correct version because of SemVer and work on fr2ir

## [4.0.5] - 2026-10-15

### Fixed
- Correct normalisation in ir2fr
- `fr2ir` cropped the first axes instead of the last ones, giving a wrong
  output with a batch of frequency responses.

### Added
- `llms.txt` at the repository root, also served at the root of the documentation.
- Agent skill `skills/udft/SKILL.md` and a Claude Code plugin marketplace
  (`.claude-plugin/marketplace.json`) to install it.
- `inshape` keyword in `fr2ir` to give the frequency grid shape, required with
  `real=True` when the last axis of the grid is odd.

## [4.0.4] - 2026-06-25

### Fixed
- Fix `laplacian` for incorrect use of dim.

## [4.0.3] - 2026-06-02

### Changed
- Conforming to [SPEC0](https://scientific-python.org/specs/spec-0000/)

## [4.0.2] - 2026-05-10

### Fixed
- readthedoc build
- fix docstring for more clairity and fix html export

## [4.0.1] - 2026-05-09

### Changed
- Small bug fix
- Doc and typo corrections

## [4.0.0] - 2026-01-22

### Changed
- Bye bye pyfftw and `lib` keyword.
- Only support `scipy` as an option.
- Use [array-api-compat](https://data-apis.org/array-api-compat/)

## [3.6.0] - 2022-08-30

### Changed
- Fix `lib` usage.
- Fix `version_info`.
- Expose `get` and `set_lib`.

### Added
- Add `valid_lib` function to have a list of valid library.

## [3.5.1] - 2022-08-02

### Changed
- Fix typing for `hnorm`.
- Remove `logging` module import
- Fix `hnorm` again
- Fix copyright date

## [3.5.0] - 2022-08-01

### Added
- Add scipy and activate parallelization by default.

### Changed
- `lib` configuration.
- Fix hnorm

## [3.4.0] - 2021-04-24

### Changed

- Change the "norm" to "hnorm" for hermitian norm computation.
- Various fix.

## [3.2.1] - 2021-04-29

### Changed

Fix diff_ir 'axe' bug.﻿

## [3.2.0] - 2021-04-24

### Changed

Numpy is the default implementation.

## [3.1.1] - 2021-04-24

### Changed

Move py.typed.

## [3.1.0] - 2021-04-24

### Added

- Add documentation.

## [3.0.1] - 2021-04-23

### Added

- You can use the library in function call.

### Changed

- Add py.typed to inform that the module is typed.
- Fix shape typing
- Change `Laplacian` to `laplacian`

## [2.0.0] - 2021-04-16

First public release.
