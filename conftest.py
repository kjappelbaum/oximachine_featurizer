# -*- coding: utf-8 -*-
"""Providing some fixtures for the tests"""

from __future__ import absolute_import
import pytest
import pandas as pd


@pytest.fixture('session')
def provide_label_dict():
    """Used to test the label parsing.

    Returns:
        dict, list -- returns a dictionary and a list of dictionaries, which is the expected parsing outcome
    """

    label_dict = {
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
        'UKUDIP01': {
            'Cu': [2],
            'Gd': [3]
        },
    }

    labels_table = [
        {
            'name': 'AZOHEC',
            'metal': 'Zn',
            'oxidationstate': 2
        },
        {
            'name': 'ACOLIP',
            'metal': 'Zn',
            'oxidationstate': 2
        },
        {
            'name': 'QAGWIG',
            'metal': 'Fe',
            'oxidationstate': 2
        },
        {
            'name': 'GOCBAD',
            'metal': 'Cu',
            'oxidationstate': 2
        },
        {
            'name': 'BUVYIB01',
            'metal': 'Fe',
            'oxidationstate': 2
        },
        {
            'name': 'GIRNIH',
            'metal': 'Cd',
            'oxidationstate': 2
        },
        {
            'name': 'FURVEU',
            'metal': 'Fe',
            'oxidationstate': 2
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Cu',
            'oxidationstate': 2
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Gd',
            'oxidationstate': 3
        },
    ]

    return label_dict, labels_table


@pytest.fixture('session')
def provide_dummy_feature_list():
    """Create a list of dictionaries that have similar structure as the featurization output.

    Returns:
        list -- list of dictionaries
    """
    feature_list = [
        {
            'name': 'AZOHEC',
            'metal': 'Zn',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'ACOLIP',
            'metal': 'Zn',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'QAGWIG',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'GOCBAD',
            'metal': 'Cu',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'BUVYIB01',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'GIRNIH',
            'metal': 'Cd',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'FURVEU',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Cu',
            'coords': [0, 0, 0],
            'feature': [2, 45567, 3564, 3574]
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Gd',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574]
        },
    ]

    return feature_list


@pytest.fixture('session')
def provide_dataframe():
    """Create a pd.DataFrame that has similar structure as the featurization output.

    Returns:
        pd.DataFrame -- mimicks the featurization output
    """
    feature_list = [
        {
            'name': 'AZOHEC',
            'metal': 'Zn',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'ACOLIP',
            'metal': 'Zn',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'QAGWIG',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'GOCBAD',
            'metal': 'Cu',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'BUVYIB01',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'GIRNIH',
            'metal': 'Cd',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'FURVEU',
            'metal': 'Fe',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Cu',
            'coords': [0, 0, 0],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
        {
            'name': 'UKUDIP01',
            'metal': 'Gd',
            'coords': [1, 1, 1],
            'feature': [2, 45567, 3564, 3574],
            'oxidationstate': 2
        },
    ]

    return pd.DataFrame(feature_list)
