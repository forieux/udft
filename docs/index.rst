========================================================
 UDFT: Unitary Discrete Fourier Transform (and related)
========================================================

|licence| |pypi| |status| |version| |maintained| |docs|

.. |licence| image:: https://img.shields.io/github/license/forieux/udft
   :alt: Documentation Status

.. |pypi| image:: https://img.shields.io/pypi/v/udft
   :alt: Pypi version

.. |status| image:: https://img.shields.io/pypi/status/udft
   :alt: Status of the code

.. |version| image:: https://img.shields.io/pypi/pyversions/udft
   :alt: Version number

.. |maintained| image:: https://img.shields.io/maintenance/yes/2026
   :alt: Maintained

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :alt: Documentation Status
   :target: https://docs.readthedocs.io/en/latest/?badge=latest

This module implements unitary (orthonormal) discrete Fourier transforms and
related functions for convolution. It is built on top of the `Array API standard
<https://data-apis.org/array-api/latest/>`_ via `array-api-compat
<https://data-apis.org/array-api-compat/>`_, making it compatible with NumPy,
PyTorch, and other compliant array libraries.

It is useful for convolution [1]: they respect the Parseval equality
:math:`\|x\|_2^2 = \|X\|_2^2`, e.g., the value of the null frequency :math:`X_0`
is equal to

.. math::

   X_0 = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} x_n, \text{ and } x_0 = \frac{1}{\sqrt{N}} \sum_{n'=0}^{N-1} X_{n'}.

and if :math:`H` is a circulant convolution with :math:`h` as a real impulse
response, then :math:`H = F^* \Lambda F` where :math:`F^*` is the unitary IDFT
computed by :func:`irdftn`, :math:`F` the unitary DFT computed by :func:`rdftn`,
and :math:`\Lambda` the frequency response computed with :func:`ir2fr` from
:math:`h`.

::

   [1] B. R. Hunt "A matrix theory proof of the discrete convolution theorem",
   IEEE Trans. on Audio and Electroacoustics, vol. au-19, no. 4, pp. 285-288,
   dec. 1971

If you are having issues, please let me know:

francois.orieux AT universite-paris-saclay.fr

Installation
============

UDFT is a single file (``udft.py``) requiring Python >= 3.12. Install with pip:

.. code-block:: sh

   pip install udft

For multithreaded FFT on NumPy arrays, install the optional SciPy dependency:

.. code-block:: sh

   pip install udft[scipy]

The project follows `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Using udft with AI agents
=========================

- `llms.txt <llms.txt>`__ follows the `llms.txt <https://llmstxt.org/>`__
  proposal and gives agents a map of this documentation.
- `skills/udft <https://github.com/forieux/udft/tree/main/skills/udft>`_ is an
  `Agent Skill <https://agentskills.io>`_ that teaches agents the conventions
  of udft (unitary normalization, last axes, ``irdftn`` shape, ``ir2fr``
  convolution model). In Claude Code, install it from the repository, which is
  also a plugin marketplace:

  .. code-block:: text

     /plugin install udft --marketplace forieux/udft

  or, in two steps, ``/plugin marketplace add forieux/udft`` then ``/plugin
  install udft@udft``. For other agents, copy the ``skills/udft`` directory
  into their skills directory.

How AI is used
==============

This project is not vibecoded. But with performant AI agents, I use them for bug
tracking and design reflection, not for code writing. I write the code, but the
one proposed by the agent is in the window on the side, obviously.

The agent and I write documentation. I reread the agent part, and the agent
reread my part, and I chose to commit. The agent writes all tests. I chose that
approach because I don't have the time to write all the documentation and tests.

I think this is a good trade-off for a project maintained by one person. It
allows me to propose a more polished version of what I publish, as if I were a
full-time developer, which is also good for my own use.

License
=======

The code is in the public domain.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   udft
