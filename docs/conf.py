# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


autodoc_mock_imports = []
try:
    import ccdc
except ImportError:
    autodoc_mock_imports.append("ccdc")


# -- Project information -----------------------------------------------------

project = "oximachine_featurizer"
copyright = "2020, Kevin Maik Jablonka, Daniele Ongari, Mohamad Moosavi, Berend Smit"
author = "Kevin Maik Jablonka, Daniele Ongari, Mohamad Moosavi, Berend Smit"

release = "0.4.1-dev"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

copybutton_selector = "div:not(.no-copy)>div.highlight pre"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "oximachine_logo.png"

# Register the theme as an extension to generate a sitemap.xml

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/kjappelbaum/oximachinerunner",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

html_show_sphinx = False

pygments_style = "sphinx"

html_use_smartypants = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
