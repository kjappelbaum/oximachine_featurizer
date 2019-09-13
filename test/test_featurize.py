# -*- coding: utf-8 -*-
"""Testing the conversion of feature files into feature matrix and label file into label vector"""

from __future__ import absolute_import
import pandas as pd
from mine_mof_oxstate.featurize import FeatureCollector


def test_make_labels_table(provide_label_dict):
    """Test conversion of raw labels in list of dictionaries"""
    label_dict, expected_table = provide_label_dict

    labels_table = FeatureCollector.make_labels_table(label_dict)

    assert labels_table == expected_table


def test_create_clean_dataframe(provide_dummy_feature_list, provide_label_dict):
    """Test merging of features and labels"""
    _, expected_table = provide_label_dict
    feature_list = provide_dummy_feature_list

    df = FeatureCollector.create_clean_dataframe(feature_list, expected_table)  # pylint:disable=invalid-name
    assert isinstance(df, pd.DataFrame)  # make sure we have a dataframe
    assert len(df) == len(expected_table)  # there are no N/As

    # an interesting case is UKUDIP01 with two metals
    assert len(df[df['name'] == 'UKUDIP01']) == 2
    assert df[(df['name'] == 'UKUDIP01') & (df['metal'] == 'Cu')]['oxidationstate'].values == 2
    assert df[(df['name'] == 'UKUDIP01') & (df['metal'] == 'Gd')]['oxidationstate'].values == 3


def test_get_x_y_names(provide_dataframe):
    """Test splitting in features, labels and names"""
    df = provide_dataframe  # pylint:disable=invalid-name

    X, y, names = FeatureCollector.get_x_y_names(df)  # pylint:disable=invalid-name

    assert len(X) == len(y) == len(names)
    # an interesting case is UKUDIP01 with two metals
