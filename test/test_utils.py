# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""Test the utils module"""
from ..utils import SymbolNameDict


def test_symbolnamedict():
    symbol_name_dict = SymbolNameDict().get_symbol_name_dict()
    assert symbol_name_dict.isinstance(dict)
    assert 'H' not in list(symbol_name_dict.keys())  # default only metals
    assert symbol_name_dict['Zn'] == 'zinc'
