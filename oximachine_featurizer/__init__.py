# -*- coding: utf-8 -*-
"""Featurization tools for the oxiMachine"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


from .featurize import FeatureCollector, GetFeatures, featurize
