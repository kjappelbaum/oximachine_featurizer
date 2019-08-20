# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""
Test the parsing class
"""
from __future__ import absolute_import
from ..parse import GetOxStatesCSD


def test_parser():
    """Test with some hand-selected MOFs"""
    test_list = ['ACIBOE', 'AZOHEC']

    expected = {'ACIBOE': {}, 'AZOHEC': {'Zn': [2]}}

    getoxstates = GetOxStatesCSD(test_list)
    result = getoxstates.run_parsing()

    assert expected == result
