# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""
Test the parsing class
"""
from __future__ import absolute_import
from mine_mof_oxstate.parse import GetOxStatesCSD
import numpy as np
import pytest


def test_parser():
    """Test with some hand-selected MOFs"""
    test_list = [
        "ACIBOE",
        "AZOHEC",
        "BADJAU",
        "ACOLIP",
        "QAGWIG",
        "GOCBAD",
        "BUVYIB01",
        "GIRNIH",
        "FURVEU",
        "GAHJUW",
    ]

    expected = {
        "ACIBOE": {"Zn": [np.nan]},
        "AZOHEC": {"Zn": [2]},
        "BADJAU": {"Sc": [np.nan]},
        "ACOLIP": {"Zn": [2]},
        "QAGWIG": {"Fe": [2]},
        "GOCBAD": {"Cu": [2]},
        "BUVYIB01": {"Fe": [2]},
        "GIRNIH": {"Cd": [2]},
        "FURVEU": {"Fe": [2]},
        "GAHJUW": {"Fe": [0]},
    }

    getoxstates = GetOxStatesCSD(test_list)
    result = getoxstates.run_parsing()

    assert expected == result


def test_parse_name(get_oxidationstate_dict):
    names, excepted = get_oxidationstate_dict
    for name, expected_res in zip(names, excepted):
        getoxstates = GetOxStatesCSD([])
        result = getoxstates.parse_name(name)
        print(result)
        assert result == expected_res
