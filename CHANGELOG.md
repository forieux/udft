# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
