# UDFT: Unitary Discrete Fourier Transform (and related)

![licence](https://img.shields.io/github/license/forieux/udft) ![pypi](https://img.shields.io/pypi/v/udft) ![status](https://img.shields.io/pypi/status/udft) ![version](https://img.shields.io/pypi/pyversions/udft) ![maintained](https://img.shields.io/maintenance/yes/2026) [![Documentation Status](https://readthedocs.org/projects/udft/badge/?version=latest)](https://udft.readthedocs.io/en/latest/?badge=latest)

This module implements unitary discrete Fourier transform, that is orthonormal
`|det(F)| = 1` and `F⁻¹ = F^*`. This module existed before the introduction of the
`norm="ortho"` keyword and is now a very (very) thin wrapper around libraries.

It is useful for convolution [1]: they respect the Parseval equality, e.g., the
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

## Using udft with AI agents

- [`llms.txt`](https://udft.readthedocs.io/en/stable/llms.txt) follows the
  [llms.txt](https://llmstxt.org/) proposal and gives agents a map of the
  documentation.
- [`skills/udft`](skills/udft/SKILL.md) is an [Agent
  Skill](https://agentskills.io) that teaches agents the conventions of udft. In
  Claude Code, install it from this repository, which is also a plugin
  marketplace:
  ```
  /plugin install udft --marketplace forieux/udft
  ```
  or with `/plugin marketplace add forieux/udft` then `/plugin install
  udft@udft`. For other agents, copy the `skills/udft` directory into their
  skills directory.

## How AI is used

This project is not vibecoded. But with performant AI agents, I use them for bug
tracking and design reflection, not for code writing. I write the code, but the
one proposed by the agent is in the window on the side, obviously.

The agent and I write documentation. I reread the agent part, and the agent
reread my part, and I chose to commit. The agent writes all tests. I chose that
approach because I don't have the time to write all the documentation and tests.

I think this is a good trade-off for a project maintained by one person. It
allows me to propose a more polished version of what I publish, as if I were a
full-time developer, which is also good for my own use.

## License

The code is in the public domain.
