# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, logging-format-interpolation, logging-fstring-interpolation, line-too-long
"""Featurization functions for the oxidation state mining project. Wrapper around matminer"""
from __future__ import absolute_import
from pathlib import Path
import os
from glob import glob
import pickle
import logging
import warnings
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.site import (
    CrystalNNFingerprint,
    CoordinationNumber,
    LocalPropertyDifference,
    BondOrientationalParameter,
    GaussianSymmFunc,
)
from .utils import read_pickle

collectorlogger = logging.getLogger('FeatureCollector')
collectorlogger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(filename)s: %(message)s', level=logging.DEBUG)
collectorlogger.addHandler(logging.FileHandler('featurecollector.log', mode='w'))


class GetFeatures:
    """Featurizer"""

    def __init__(self, structure, outpath):
        """Generates features for a list of structures

        Args:
            structure_paths (str): path to structure
            outpath (str): path to which the features will be dumped

        Returns:

        """
        featurizelogger = logging.getLogger('Featurize')
        featurizelogger.setLevel(logging.DEBUG)
        logging.basicConfig(
            filename='featurize.log',
            format='%(filename)s: %(message)s',
            level=logging.DEBUG,
        )

        self.outpath = outpath
        self.logger = featurizelogger
        self.path = structure
        self.structure = None
        self.metal_sites = []
        self.metal_indices = []
        self.features = defaultdict(dict)
        self.outname = os.path.join(self.outpath, ''.join([Path(structure).stem, '.pkl']))

    def precheck(self):
        """Fail early

        Returns:
            bool: True if check ok (if pymatgen can load structure)

        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                atoms = read(self.path)
                self.structure = AseAtomsAdaptor.get_structure(atoms)  # ase parser is a bit more robust
                return True
            except Exception:  # pylint: disable=broad-except
                return False

    def get_metal_sites(self):
        """Stores all metal sites of structure  to list"""
        for idx, site in enumerate(self.structure):
            if site.species.elements[0].is_metal:
                self.metal_sites.append(site)
                self.metal_indices.append(idx)

    def get_feature_vectors(self, site):
        """Runs matminer on one site"""
        featurizer = MultipleFeaturizer([
            CrystalNNFingerprint.from_preset('ops'),
            CoordinationNumber(),
            LocalPropertyDifference.from_preset('ward-prb-2017'),
            BondOrientationalParameter(),
            GaussianSymmFunc(),
        ])

        X = featurizer.featurize(self.structure, site)
        return X

    def dump_features(self):
        """Dumps all the features into one pickle file"""
        with open(self.outname, 'wb') as filehandle:
            pickle.dump(dict(self.features), filehandle)

    def run_featurization(self):
        """loops over sites if check ok"""
        if self.precheck():
            self.get_metal_sites()
            try:
                for idx, metal_site in enumerate(self.metal_sites):
                    self.features[metal_site.species_string]['feature'] = self.get_feature_vectors(
                        self.metal_indices[idx])
                    self.features[metal_site.species_string]['coords'] = metal_site.coords
                self.dump_features()
            except Exception:  # pylint: disable=broad-except
                self.logger.error('could not featurize {}'.format(self.path))
        else:
            self.logger.error('could not load {}'.format(self.path))


class FeatureCollector:
    """convert features from a folder of pickle files to three
    pickle files for feature matrix, label vector and names list. """

    def __init__(
            self,
            inpath: str = None,
            labelpath: str = None,
            outdir_labels: str = 'data/labels',
            outdir_features: str = 'data/features',
            outdir_helper: str = 'data/helper',
    ):
        self.inpath = inpath
        self.labelpath = labelpath
        self.outdir_labels = outdir_labels
        self.outdir_features = outdir_features
        self.outdir_helper = outdir_helper

        self.picklefiles = glob(os.path.join(inpath, '*.pkl'))
        self.forbidden_list = list(
            read_pickle(
                '/home/kevin/Dropbox/proj62_guess_oxidation_states/machine_learn_oxstates/data/helper/two_ox_states.pkl'
            ))
        collectorlogger.info(
            f'initialized feature collector: {len(self.forbidden_list)} forbidden structures, {len(self.picklefiles)} files with features'
        )

    def _featurecollection(self) -> Tuple[np.array, np.array, list]:
        """
        Runs the feature collection workflow.

        Returns:
            Tuple[np.array, np.array, list] -- numpy arrays of features and labels and list of names
        """
        feature_list = FeatureCollector.create_feature_list(self.picklefiles, self.forbidden_list)
        label_raw = read_pickle(self.labelpath)
        label_list = FeatureCollector.make_labels_table(label_raw)
        df = FeatureCollector.create_clean_dataframe(feature_list, label_list)
        x, y, names = FeatureCollector.get_x_y_names(df)
        return x, y, names

    def dump_featurecollection(self) -> None:
        """Collect features and write features, labels and names to seperate files

        Returns:
            None -- [description]
        """
        x, y, names = self._featurecollection()
        FeatureCollector.write_output(x, y, names, self.outdir_labels, self.outdir_features, self.outdir_helper)

    def return_featurecollection(self) -> Tuple[np.array, np.array, list]:
        x, y, names = self._featurecollection()
        return x, y, names

    @staticmethod
    def create_feature_list(picklefiles: list, forbidden_list: list) -> list:
        """Reads a list of pickle files into dictionary

        Arguments:
            picklefiles {list} -- list of paths
            forbidden_list {list} -- list of forbidden names (CSD naming convention)

        Returns:
            list -- parsed pickle contents
        """
        collectorlogger.info('reading pickle files with features')
        result_list = []
        if not isinstance(forbidden_list, list):
            forbidden_list = []

        for pickle_file in picklefiles:
            if Path(pickle_file).stem not in forbidden_list:
                result_list.extend(FeatureCollector.create_dict_for_feature_table(pickle_file))
            else:
                collectorlogger.info(f'{pickle_file} is in forbidden list and will not be considered for X, y, names')
        return result_list

    @staticmethod
    def make_labels_table(raw_labels: dict) -> list:
        """Read raw labeling output into a dictionary format that can be used to construct pd.DataFrames

        Warning: assumes that each metal in the structure has the same oxidation states as it takes the first
        list element. Cases in which this is not fulfilled need to be filtered out earlier.

        Arguments:
            raw_labels {dict} -- nested dictionary of {name: {metal: [oxidationstates]}}

        Returns:
            list -- list of dictionaries of the form [{'name':, 'metal':, 'oxidationstate':}]
        """
        collectorlogger.info('converting raw list of features into list of site dictionaries')
        result_list = []
        for key, value in raw_labels.items():
            for metal, oxstate in value.items():
                result_list.append({'name': key, 'metal': metal, 'oxidationstate': oxstate[0]})
        collectorlogger.info(f'collected {len(result_list)} features')
        return result_list

    @staticmethod
    def create_clean_dataframe(feature_list: list, label_list: list) -> pd.DataFrame:
        """Merge the features and the labels on names and metals and drop entry rows

        Arguments:
            feature_list {list} --  list of dictionaries of the features, names and metals
            label_list {list} -- list of dicionaries with names, metals and labels

        Returns:
            pd.DataFrame -- Dataframe with each row describing a seperate metal site
        """
        collectorlogger.info('merging labels and features')
        df_features = pd.DataFrame(feature_list)
        df_labels = pd.DataFrame(label_list)

        df_merged = pd.merge(
            df_features,
            df_labels,
            left_on=['name', 'metal'],
            right_on=['name', 'metal'],
        )
        df_merged.dropna(inplace=True)
        return df_merged

    @staticmethod
    def get_x_y_names(df: pd.DataFrame) -> Tuple[np.array, np.array, list]:
        """Splits the columns of a dataframe into features, labels and names

        Arguments:
            df {pd.DataFrame} -- dataframe that in each row contains a full description of
                a metal site with feature, oxidation state, metal  and name

        Returns:
            Tuple[np.array, np.array, list] -- [description]
        """
        names = list(df['name'])
        features = np.array(df['feature'])
        labels = np.array(df['oxidationstate'])
        return features, labels, names

    @staticmethod
    def create_dict_for_feature_table(picklefile: str) -> list:
        """Reads in a pickle with features and returns a list of dictionaries with one dictionary per metal site.

        Arguments:
            picklefile {str} -- path to pickle file

        Returns:
            list -- list of dicionary
        """
        result = read_pickle(picklefile)

        result_list = []
        for key, value in result.items():
            result_dict = {
                'metal': key,
                'coords': value['coords'],
                'feature': value['feature'],
                'name': Path(picklefile).stem,
            }

            result_list.append(result_dict)

        return result_list

    @staticmethod
    def write_output(
            x: np.array,
            y: np.array,
            names: list,
            outdir_labels: str,
            outdir_features: str,
            outdir_helper: str,
    ) -> None:
        """writes feature array, label array and name array into output files in outdir/datetime/{x,y}.npy and outdir/datetime/names.pkl

        Arguments:
            x {np.array} -- feature matrix
            y {np.array} -- label vector
            names {list} -- name list (csd  identifiers)
            outdir_labels {str} -- directory into which labels are written
            outdir_features {str} -- directory into which features are written
            outdir_helper {str} -- directory into which names are written

        Returns:
            None --
        """
        # timestr = time.strftime('%Y%m%d-%H%M%S')
        # outpath_base = os.path.join(outdir, timestr)

        np.save(os.path.join(outdir_labels, 'features'), x)
        np.save(os.path.join(outdir_features, 'labels'), y)

        with open(os.path.join(outdir_helper, 'names.pkl'), 'wb') as picklefile:
            pickle.dump(names, picklefile)

        collectorlogger.info('stored features into labels and names into')
