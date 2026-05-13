"""Sphinx configuration for telescope-sim."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the source available so autodoc can import it
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import telescope_sim

# -- Project info -----------------------------------------------------------

project = "telescope-sim"
author = "Ian Cunnyngham"
copyright = "2026, Ian Cunnyngham"
release = telescope_sim.__version__
version = ".".join(release.split(".")[:2])

# -- General config ---------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.automodapi",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Napoleon (numpy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "hcipy": ("https://docs.hcipy.org/stable/", None),
}

# nbsphinx: notebooks ship pre-executed; do not re-run on build
nbsphinx_execute = "never"

# Auto-docstring style
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- HTML output ------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"telescope-sim {release}"
