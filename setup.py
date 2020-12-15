#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, missing-docstring

import io
import os

from setuptools import find_packages, setup

import versioneer

# Package meta-data.
NAME = "oximachine_featurizer"
DESCRIPTION = "Mine MOF oxidation states and featurize metal sites."
URL = "https://github.com/kjappelbaum/oximachine_featurizer"
EMAIL = "kevin.jablonka@epfl.ch"
AUTHOR = "Kevin M. Jablonka, Daniele Ongari, Berend Smit"
REQUIRES_PYTHON = ">=3.6.0"

# What packages are required for this module to be executed?
with open("requirements.txt") as fh:
    REQUIRED = fh.readlines()

# What packages are optional?
EXTRAS = {
    "dev": [
        "pre-commit~=2.6.0",
        "pylint~=2.5.3",
        "pytest~=6.0.1",
        "versioneer~=0.18",
        "isort~=4.3.21",
        "black",
    ],
    "docs": [
        "sphinx~=3.3.1",
        "sphinx-book-theme~=0.0.39",
        "sphinx-autodoc-typehints~=1.11.1",
        "sphinx-copybutton~=0.3.1",
    ],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    entry_points={
        "console_scripts": [
            "run_parsing=run.run_parsing:main",
            "run_parsing_reference=run.run_parsing_reference:main",
            "run_mine_mp=run.run_mine_mp:main",
            "run_featurization=run.run_featurization:main",
        ]
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
