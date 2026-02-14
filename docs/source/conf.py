"""Sphinx configuration for LipiDetective documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as get_version
from pathlib import Path

# Add source directory to path for autodoc
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# -- Project information -------------------------------------------------------

project = "LipiDetective"
copyright = "2026, Vivian Wuerf"  # noqa: A001
author = "Vivian Wuerf"
try:
    release = get_version("lipidetective")
except PackageNotFoundError:
    release = "0.0.0-dev"

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_togglebutton",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    # Version selector requires ReadTheDocs hosting; disable for GitHub Pages.
    # See https://github.com/readthedocs/sphinx_rtd_theme/issues/1624
    "version_selector": False,
}
html_static_path = ["_static"]

# -- Extension configuration ---------------------------------------------------

add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable", None),
}

autodoc_member_order = "bysource"
