# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""Test the utils module"""
from __future__ import absolute_import

from oximachine_featurizer.utils import SymbolNameDict


def test_symbolnamedict():
    symbol_name_dict = SymbolNameDict().get_symbol_name_dict()
    assert isinstance(symbol_name_dict, dict)
    assert 'H' not in list(symbol_name_dict.keys())  # default only metals
    assert symbol_name_dict['Zn'] == 'zinc'
