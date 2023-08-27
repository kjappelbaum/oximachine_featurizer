# -*- coding: utf-8 -*-
"""Test the utils module"""
from oximachine_featurizer.utils import SymbolNameDict


def test_symbolnamedict():
    symbol_name_dict = SymbolNameDict().get_symbol_name_dict()
    assert isinstance(symbol_name_dict, dict)
    assert "H" not in list(symbol_name_dict.keys())  # default only metals
    assert symbol_name_dict["Zn"] == "zinc"
