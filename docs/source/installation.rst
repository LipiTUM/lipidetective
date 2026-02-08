Installation
============

.. _installation:

Requirements
------------

- Python 3.11 or later
- CUDA-capable GPU (optional, but recommended for training)

Install with UV (Recommended)
-----------------------------

`UV <https://docs.astral.sh/uv/>`_ is the recommended package manager:

.. code-block:: console

   $ uv sync

This installs LipiDetective and all runtime dependencies.

Install from Source with pip
----------------------------

Since LipiDetective is not yet published on PyPI, install from the cloned
repository:

.. code-block:: console

   $ pip install .

Development Setup
-----------------

To install with all development and documentation tools:

.. code-block:: console

   $ uv sync --all-groups

This includes ``pytest``, ``ruff``, ``mypy``, Sphinx, and all other
development dependencies.

You can also install individual groups:

.. code-block:: console

   $ uv sync --group dev    # Testing, linting, type checking
   $ uv sync --group docs   # Documentation tools only

Verify Installation
-------------------

.. code-block:: console

   $ uv run lipidetective --help
