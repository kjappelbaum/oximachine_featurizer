# -*- coding: utf-8 -*-
# pylint:disable=invalid-name, logging-format-interpolation
"""Featurization functions for the oxidation state mining project. Wrapper around matminer"""
from __future__ import absolute_import
from pathlib import Path
import os
import pickle
import logging
from pymatgen import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.site import (CrystalNNFingerprint, CoordinationNumber, LocalPropertyDifference,
                                       BondOrientationalParameter, GaussianSymmFunc)


class GetFeatures():
    """Featurizer"""

    def __init__(self, structure, outpath):
        """Generates features for a list of structures

        Args:
            structure_paths (str): path to structure
            outpath (str): path to which the features will be dumped

        Returns:

        """
        self.outpath = outpath
        logging.basicConfig(filename=os.path.join(self.outpath, 'log'),
                            level=logging.DEBUG,
                            format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                            datefmt='%H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.path = structure
        self.structure = None
        self.metal_sites = []
        self.features = {}
        self.outname = os.path.join(self.outpath, ''.join([Path(structure).stem, '.pkl']))

    def precheck(self):
        """Fail early

        Returns:
            bool: True if check ok (if pymatgen can load structure)

        """
        try:
            self.structure = Structure.from_file(self.path)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    def get_metal_sites(self):
        """Stores all metal sites of structure  to list"""
        for site in self.structure:
            if site.species.elements[0].is_metal:
                self.metal_sites.append(site)

    def get_feature_vectors(self, site):
        """Runs matminer on one site"""
        featurizer = MultipleFeaturizer([
            CrystalNNFingerprint, CoordinationNumber, LocalPropertyDifference, BondOrientationalParameter,
            GaussianSymmFunc
        ])

        X = featurizer.featurize(site, self.structure)
        return X

    def dump_features(self):
        """Dumps all the features into one pickle file"""
        with open(self.outname, 'wb') as filehandle:
            pickle.dump(self.features, filehandle)

    def log_error(self):
        """logs error message"""
        self.logger.error('could not load {}'.format(self.path))

    def run_featurization(self):
        """loops over sites if check ok"""
        if self.precheck():
            self.get_metal_sites()
            for metal_site in self.metal_sites:
                self.features[metal_site.species_string]['feature'] = self.get_feature_vectors(metal_site)
                self.features[metal_site.species_string]['coords'] = metal_site.coords

            self.dump_features()
        else:
            self.log_error()
