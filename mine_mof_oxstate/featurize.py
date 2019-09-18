# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, logging-format-interpolation, logging-fstring-interpolation, line-too-long, dangerous-default-value
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

FEATURE_RANGES_DICT = {
    'crystal_nn_fingerprint': (0, 61),
    'cn': (61, 62),
    'ward_prd': (62, 84),
    'bond_orientational': (84, 94),
    'behler_parinello': (94, 103),
}

FEATURE_LABELS_ALL = [
    'wt CN_1',
    'sgl_bd CN_1',
    'wt CN_2',
    'L-shaped CN_2',
    'water-like CN_2',
    'bent 120 degrees CN_2',
    'bent 150 degrees CN_2',
    'linear CN_2',
    'wt CN_3',
    'trigonal planar CN_3',
    'trigonal non-coplanar CN_3',
    'T-shaped CN_3',
    'wt CN_4',
    'square co-planar CN_4',
    'tetrahedral CN_4',
    'rectangular see-saw-like CN_4',
    'see-saw-like CN_4',
    'trigonal pyramidal CN_4',
    'wt CN_5',
    'pentagonal planar CN_5',
    'square pyramidal CN_5',
    'trigonal bipyramidal CN_5',
    'wt CN_6',
    'hexagonal planar CN_6',
    'octahedral CN_6',
    'pentagonal pyramidal CN_6',
    'wt CN_7',
    'hexagonal pyramidal CN_7',
    'pentagonal bipyramidal CN_7',
    'wt CN_8',
    'body-centered cubic CN_8',
    'hexagonal bipyramidal CN_8',
    'wt CN_9',
    'q2 CN_9',
    'q4 CN_9',
    'q6 CN_9',
    'wt CN_10',
    'q2 CN_10',
    'q4 CN_10',
    'q6 CN_10',
    'wt CN_11',
    'q2 CN_11',
    'q4 CN_11',
    'q6 CN_11',
    'wt CN_12',
    'cuboctahedral CN_12',
    'q2 CN_12',
    'q4 CN_12',
    'q6 CN_12',
    'wt CN_13',
    'wt CN_14',
    'wt CN_15',
    'wt CN_16',
    'wt CN_17',
    'wt CN_18',
    'wt CN_19',
    'wt CN_20',
    'wt CN_21',
    'wt CN_22',
    'wt CN_23',
    'wt CN_24',
    'CN_VoronoiNN',
    'local difference in Number',
    'local difference in MendeleevNumber',
    'local difference in AtomicWeight',
    'local difference in MeltingT',
    'local difference in Column',
    'local difference in Row',
    'local difference in CovalentRadius',
    'local difference in Electronegativity',
    'local difference in NsValence',
    'local difference in NpValence',
    'local difference in NdValence',
    'local difference in NfValence',
    'local difference in NValence',
    'local difference in NsUnfilled',
    'local difference in NpUnfilled',
    'local difference in NdUnfilled',
    'local difference in NfUnfilled',
    'local difference in NUnfilled',
    'local difference in GSvolume_pa',
    'local difference in GSbandgap',
    'local difference in GSmagmom',
    'local difference in SpaceGroupNumber',
    'BOOP Q l=1',
    'BOOP Q l=2',
    'BOOP Q l=3',
    'BOOP Q l=4',
    'BOOP Q l=5',
    'BOOP Q l=6',
    'BOOP Q l=7',
    'BOOP Q l=8',
    'BOOP Q l=9',
    'BOOP Q l=10',
    'G2_0.05',
    'G2_4.0',
    'G2_20.0',
    'G2_80.0',
    'G4_0.005_1.0_1.0',
    'G4_0.005_1.0_-1.0',
    'G4_0.005_4.0_1.0',
    'G4_0.005_4.0_-1.0',
]


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
        self.featurizer = MultipleFeaturizer([
            CrystalNNFingerprint.from_preset('ops'),
            CoordinationNumber(),
            LocalPropertyDifference.from_preset('ward-prb-2017'),
            BondOrientationalParameter(),
            GaussianSymmFunc(),
        ])

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

        X = self.featurizer.featurize(self.structure, site)

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

    def __init__(  # pylint:disable=too-many-arguments
            self,
            inpath: str = None,
            labelpath: str = None,
            outdir_labels: str = 'data/labels',
            outdir_features: str = 'data/features',
            outdir_helper: str = 'data/helper',
            forbidden_picklepath:
            str = '/home/kevin/Dropbox/proj62_guess_oxidation_states/machine_learn_oxstates/data/helper/two_ox_states.pkl',
            exclude_dir: str = '/home/kevin/Dropbox (LSMO)/proj62_guess_oxidation_states/test_structures/showcases',
            selected_features: list = [
                'crystal_nn_fingerprint',
                'ward_prd',
                'bond_orientational',
                'behler_parinello',
            ],
    ):
        """Initializes a feature collector.

        WARNING! The fingerprint selection function assumes that the full feature vector in the
        pickle files has the columns as specified in FEATURE_LABELS_ALL

        Keyword Arguments:
            inpath {str} -- path to directory with one pickle file per structure (default: {None})
            labelpath {str} -- path to picklefile with labels (default: {None})
            outdir_labels {str} -- path to output directory for labelsfile (default: {"data/labels"})
            outdir_features {str} -- path to output directory for featuresfile (default: {"data/features"})
            outdir_helper {str} -- path to output directory for helper files (feature names, structure names) (default: {"data/helper"})
            forbidden_picklepath {str} -- path to picklefile with list of forbidden CSD names (default: {"/home/kevin/Dropbox/proj62_guess_oxidation_states/machine_learn_oxstates/data/helper/two_ox_states.pkl"})
            exclude_dir {str} -- path to directory with structure names are forbidden as well (default: {"/home/kevin/Dropbox (LSMO)/proj62_guess_oxidation_states/test_structures/showcases"})
            selected_features {list} -- list of selected features. Available crystal_nn_fingerprint, cn, ward_prb, bond_orientational, behler_parinello
              (default: {["crystal_nn_fingerprint","ward_prd","bond_orientational","behler_parinello",]})
        """
        self.inpath = inpath
        self.labelpath = labelpath
        self.outdir_labels = outdir_labels
        self.outdir_features = outdir_features
        self.outdir_helper = outdir_helper
        self.selected_features = selected_features

        for feature in self.selected_features:
            assert feature in list(FEATURE_RANGES_DICT.keys())

        self.picklefiles = glob(os.path.join(inpath, '*.pkl'))
        self.forbidden_list = (list(read_pickle(forbidden_picklepath)) if forbidden_picklepath is not None else [])
        if exclude_dir is not None:
            all_to_exclude = [Path(p).stem for p in glob(os.path.join(exclude_dir, '*.cif'))]
            self.forbidden_list.extend(all_to_exclude)

        collectorlogger.info(
            f'initialized feature collector: {len(self.forbidden_list)} forbidden structures, {len(self.picklefiles)} files with features'
        )

    def _select_features(self, X):
        """Selects the feature and dumps the names as pickle in the helper directory"""
        to_hstack = []
        featurenames = []
        for feature in self.selected_features:
            lower, upper = FEATURE_RANGES_DICT[feature]
            to_hstack.append(X[lower:upper])
            featurenames.extend(FEATURE_LABELS_ALL[lower:upper])

        with open(os.path.join(self.outdir_helper, 'feature_names.pkl'), 'wb') as fh:
            pickle.dump(featurenames, fh)
        return np.hstack(to_hstack)

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
        x_selected = self._select_features(x)
        return x_selected, y, names

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
    def _partial_match_in_name(name: str, forbidden_list: list) -> bool:
        """Tries to match also partial names, e.g. to ensure that  MAHSUK01 or
        MAHSUK02 is also matched when only MAHSUK is in the forbidden list"""
        return any(name.rstrip('1234567890') in s for s in forbidden_list)

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
            if not FeatureCollector._partial_match_in_name(Path(pickle_file).stem, forbidden_list):
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
        feature_list = [l for l in df['feature']]
        features = np.array(feature_list)
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

        np.save(os.path.join(outdir_features, 'features'), x)
        np.save(os.path.join(outdir_labels, 'labels'), y)

        with open(os.path.join(outdir_helper, 'names.pkl'), 'wb') as picklefile:
            pickle.dump(names, picklefile)

        collectorlogger.info('stored features into labels and names into')
