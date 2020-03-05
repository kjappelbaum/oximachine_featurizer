# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""
Test the parsing class
"""
from __future__ import absolute_import
from mine_mof_oxstate.parse import GetOxStatesCSD


def test_parser():
    """Test with some hand-selected MOFs"""
    test_list = [
        'ACIBOE',
        'AZOHEC',
        'BADJAU',
        'ACOLIP',
        'QAGWIG',
        'GOCBAD',
        'BUVYIB01',
        'GIRNIH',
        'FURVEU',
    ]

    expected = {
        'ACIBOE': {},
        'AZOHEC': {
            'Zn': [2]
        },
        'BADJAU': {},
        'ACOLIP': {
            'Zn': [2]
        },
        'QAGWIG': {
            'Fe': [2]
        },
        'GOCBAD': {
            'Cu': [2]
        },
        'BUVYIB01': {
            'Fe': [2]
        },
        'GIRNIH': {
            'Cd': [2]
        },
        'FURVEU': {
            'Fe': [2]
        },
        'GAHJUW': {
            'Fe': [0]
        },
    }

    getoxstates = GetOxStatesCSD(test_list)
    result = getoxstates.run_parsing()

    assert expected == result
