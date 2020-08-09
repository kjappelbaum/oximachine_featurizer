# -*- coding: utf-8 -*-
import copy

import numpy as np
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.site import (LocalStructOrderParams, cn_motif_op_params, cn_target_motif_op)
from matminer.utils.caching import get_nearest_neighbors
from matminer.utils.data import MagpieData
from pymatgen.analysis.local_env import VoronoiNN

from .crystalnn import CrystalNN


class LocalPropertyStatsNew(BaseFeaturizer):
    """
    Differences, minima and maxima in elemental properties between site and its neighboring sites.
    Uses the Voronoi tessellation of the structure to determine the
    neighbors of the site, and assigns each neighbor (:math:`n`) a
    weight (:math:`A_n`) that corresponds to the area of the facet
    on the tessellation corresponding to that neighbor.
    The local property difference is then computed by
    :math:`\\frac{\sum_n {A_n |p_n - p_0|}}{\sum_n {A_n}}`
    where :math:`p_n` is the property (e.g., atomic number) of a neighbor
    and :math:`p_0` is the property of a site. If signed parameter is assigned
    True, signed difference of the properties is returned instead of absolute
    difference.
    Features:
        - "local property stat in [property]"
    References:
         `Ward et al. _PRB_ 2017 <http://link.aps.org/doi/10.1103/PhysRevB.96.024104>`_
    """

    def __init__(self, data_source=MagpieData(), weight='area', properties=('Electronegativity',)):
        """ Initialize the featurizer
        Args:
            data_source (AbstractData) - Class from which to retrieve
                elemental properties
            weight (str) - What aspect of each voronoi facet to use to
                weigh each neighbor (see VoronoiNN)
            properties ([str]) - List of properties to use (default=['Electronegativity'])
            signed (bool) - whether to return absolute difference or signed difference of
                            properties(default=False (absolute difference))
        """
        self.data_source = data_source
        self.properties = properties
        self.weight = weight

    @staticmethod
    def from_preset(preset):
        """
        Create a new LocalPropertyStats class according to a preset
        Args:
            preset (str) - Name of preset
        """

        if preset == 'interpretable':
            return LocalPropertyStatsNew(
                data_source=MagpieData(),
                properties=[
                    'MendeleevNumber',
                    'Column',
                    'Row',
                    'Electronegativity',
                    'NsValence',
                    'NpValence',
                    'NdValence',
                    'NfValence',
                    'NValence',
                    'NsUnfilled',
                    'NpUnfilled',
                    'NdUnfilled',
                    'NfUnfilled',
                    'NUnfilled',
                    'GSbandgap',
                ],
            )
        else:
            raise ValueError('Unrecognized preset: ' + preset)

    def featurize(self, strc, idx):
        # Get the targeted site
        my_site = strc[idx]

        # Get the tessellation of a site
        nn = get_nearest_neighbors(
            VoronoiNN(weight=self.weight, cutoff=8, compute_adj_neighbors=False),
            strc,
            idx,
        )

        # Get the element and weight of each site
        elems = [n['site'].specie for n in nn]
        weights = [n['weight'] for n in nn]

        # Compute the difference for each property
        output = np.zeros((len(self.properties),))
        output_signed = np.zeros((len(self.properties),))
        output_max = np.zeros((len(self.properties),))
        output_min = np.zeros((len(self.properties),))

        total_weight = np.sum(weights)
        for i, p in enumerate(self.properties):
            my_prop = self.data_source.get_elemental_property(my_site.specie, p)
            n_props = self.data_source.get_elemental_properties(elems, p)
            output[i] = (np.dot(weights, np.abs(np.subtract(n_props, my_prop))) / total_weight)
            output_signed[i] = (np.dot(weights, np.subtract(n_props, my_prop)) / total_weight)
            output_max[i] = np.max(np.subtract(n_props, my_prop))
            output_min[i] = np.min(np.subtract(n_props, my_prop))
        return np.hstack([output, output_signed, output_max, output_min])

    def feature_labels(self):

        return (['local difference in ' + p for p in self.properties] +
                ['local signed difference in ' + p for p in self.properties] +
                ['maximum local difference in ' + p for p in self.properties] +
                ['minimum local difference in ' + p for p in self.properties])

    def citations(self):
        return [
            '@article{Ward2017,'
            'author = {Ward, Logan and Liu, Ruoqian '
            'and Krishna, Amar and Hegde, Vinay I. '
            'and Agrawal, Ankit and Choudhary, Alok '
            'and Wolverton, Chris},'
            'doi = {10.1103/PhysRevB.96.024104},'
            'journal = {Physical Review B},'
            'pages = {024104},'
            'title = {{Including crystal structure attributes '
            'in machine learning models of formation energies '
            'via Voronoi tessellations}},'
            'url = {http://link.aps.org/doi/10.1103/PhysRevB.96.014107},'
            'volume = {96},year = {2017}}',
            '@article{jong_chen_notestine_persson_ceder_jain_asta_gamst_2016,'
            'title={A Statistical Learning Framework for Materials Science: '
            'Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds}, '
            'volume={6}, DOI={10.1038/srep34256}, number={1}, journal={Scientific Reports}, '
            'author={Jong, Maarten De and Chen, Wei and Notestine, Randy and Persson, '
            'Kristin and Ceder, Gerbrand and Jain, Anubhav and Asta, Mark and Gamst, Anthony}, '
            'year={2016}, month={Mar}}',
        ]

    def implementors(self):
        return ['Logan Ward', 'Aik Rui Tan', 'Kevin Jablonka']


class CrystalNNFingerprint(BaseFeaturizer):
    """
    A local order parameter fingerprint for periodic crystals.
    The fingerprint represents the value of various order parameters for the
    site. The "wt" order parameter describes how consistent a site is with a
    certain coordination number. The remaining order parameters are computed
    by multiplying the "wt" for that coordination number with the OP value.
    The chem_info parameter can be used to also get chemical descriptors that
    describe differences in some chemical parameter (e.g., electronegativity)
    between the central site and the site neighbors.
    """

    @staticmethod
    def from_preset(preset, **kwargs):
        """
        Use preset parameters to get the fingerprint
        Args:
            preset (str): name of preset ("cn" or "ops")
            **kwargs: other settings to be passed into CrystalNN class
        """
        if preset == 'cn':
            op_types = {k + 1: ['wt'] for k in range(24)}
            return CrystalNNFingerprint(op_types, **kwargs)

        elif preset == 'ops':
            op_types = copy.deepcopy(cn_target_motif_op)
            for k in range(24):
                if k + 1 in op_types:
                    op_types[k + 1].insert(0, 'wt')
                else:
                    op_types[k + 1] = ['wt']

            return CrystalNNFingerprint(op_types, chem_info=None, **kwargs)

        else:
            raise RuntimeError('preset "{}" is not supported in ' 'CrystalNNFingerprint'.format(preset))

    def __init__(self, op_types, chem_info=None, **kwargs):
        """
        Initialize the CrystalNNFingerprint. Use the from_preset() function to
        use default params.
        Args:
            op_types (dict): a dict of coordination number (int) to a list of str
                representing the order parameter types
            chem_info (dict): a dict of chemical properties (e.g., atomic mass)
                to dictionaries that map an element to a value
                (e.g., chem_info["Pauling scale"]["O"] = 3.44)
            **kwargs: other settings to be passed into CrystalNN class
        """

        self.op_types = copy.deepcopy(op_types)
        self.cnn = CrystalNN(**kwargs)
        if chem_info is not None:
            self.chem_info = copy.deepcopy(chem_info)
            self.chem_props = list(chem_info.keys())
        else:
            self.chem_info = None

        self.ops = {}  # load order parameter objects & paramaters
        for cn, t_list in self.op_types.items():
            self.ops[cn] = []
            for t in t_list:
                if t == 'wt':
                    self.ops[cn].append(t)
                else:
                    ot = t
                    p = None
                    if cn in cn_motif_op_params.keys():
                        if t in cn_motif_op_params[cn].keys():
                            ot = cn_motif_op_params[cn][t][0]
                            if len(cn_motif_op_params[cn][t]) > 1:
                                p = cn_motif_op_params[cn][t][1]
                    self.ops[cn].append(LocalStructOrderParams([ot], parameters=[p]))

    def featurize(self, struct, idx):
        """
        Get crystal fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            list of weighted order parameters of target site.
        """

        nndata = self.cnn.get_nn_data(struct, idx)
        max_cn = sorted(self.op_types)[-1]

        cn_fingerprint = []

        if self.chem_info is not None:
            prop_delta = {}  # dictionary of chemical property to final value
            for prop in self.chem_props:
                prop_delta[prop] = 0
            sum_wt = 0
            elem_central = struct.sites[idx].specie.symbol
            specie_central = str(struct.sites[idx].specie)

        for k in range(max_cn):
            cn = k + 1
            wt = nndata.cn_weights.get(cn, 0)
            if cn in self.ops:
                for op in self.ops[cn]:
                    if op == 'wt':
                        cn_fingerprint.append(wt)

                        if self.chem_info is not None and wt != 0:
                            # Compute additional chemistry-related features
                            sum_wt += wt
                            neigh_sites = [d['site'] for d in nndata.cn_nninfo[cn]]

                            for prop in self.chem_props:
                                # get the value for specie, if not fall back to
                                # value defined for element
                                prop_central = self.chem_info[prop].get(
                                    specie_central,
                                    self.chem_info[prop].get(elem_central),
                                )

                                for neigh in neigh_sites:
                                    elem_neigh = neigh.specie.symbol
                                    specie_neigh = str(neigh.specie)
                                    prop_neigh = self.chem_info[prop].get(
                                        specie_neigh,
                                        self.chem_info[prop].get(elem_neigh),
                                    )

                                    prop_delta[prop] += (wt * (prop_neigh - prop_central) / cn)

                    elif wt == 0:
                        cn_fingerprint.append(wt)
                    else:
                        neigh_sites = [d['site'] for d in nndata.cn_nninfo[cn]]
                        opval = op.get_order_parameters(
                            [struct[idx]] + neigh_sites,
                            0,
                            indices_neighs=list(range(1,
                                                      len(neigh_sites) + 1)),
                        )[0]
                        opval = opval or 0  # handles None
                        cn_fingerprint.append(wt * opval)
        chem_fingerprint = []

        if self.chem_info is not None:
            for val in prop_delta.values():
                chem_fingerprint.append(val / sum_wt)

        return cn_fingerprint + chem_fingerprint

    def feature_labels(self):
        labels = []
        max_cn = sorted(self.op_types)[-1]
        for k in range(max_cn):
            cn = k + 1
            if cn in list(self.ops.keys()):
                for op in self.op_types[cn]:
                    labels.append('{} CN_{}'.format(op, cn))
        if self.chem_info is not None:
            for prop in self.chem_props:
                labels.append('{} local diff'.format(prop))
        return labels

    def citations(self):
        return []

    def implementors(self):
        return ['Anubhav Jain', 'Nils E.R. Zimmermann']
