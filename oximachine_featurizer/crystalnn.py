# -*- coding: utf-8 -*-
import math
from collections import namedtuple

import numpy as np
from numba import jit
from pymatgen.analysis.local_env import (NearNeighbors, VoronoiNN, _get_default_radius, _get_radius)


class CrystalNN(NearNeighbors):
    """
    This is custom near neighbor method intended for use in all kinds of
    periodic structures (metals, minerals, porous structures, etc). It is based
    on a Voronoi algorithm and uses the solid angle weights to determine the
    probability of various coordination environments. The algorithm can also
    modify probability using smooth distance cutoffs as well as Pauling
    electronegativity differences. The output can either be the most probable
    coordination environment or a weighted list of coordination environments.
    """

    NNData = namedtuple('nn_data', ['all_nninfo', 'cn_weights', 'cn_nninfo'])

    def __init__(
        self,
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.5, 1),
        x_diff_weight=3.0,
        porous_adjustment=True,
        search_cutoff=8,
        fingerprint_length=None,
    ):
        """
        Initialize CrystalNN with desired parameters. Default parameters assume
        "chemical bond" type behavior is desired. For geometric neighbor
        finding (e.g., structural framework), set (i) distance_cutoffs=None,
        (ii) x_diff_weight=0.0 and (optionally) (iii) porous_adjustment=False
        which will disregard the atomic identities and perform best for a purely
        geometric match.
        Args:
            weighted_cn: (bool) if set to True, will return fractional weights
                for each potential near neighbor.
            cation_anion: (bool) if set True, will restrict bonding targets to
                sites with opposite or zero charge. Requires an oxidation states
                on all sites in the structure.
            distance_cutoffs: ([float, float]) - if not None, penalizes neighbor
                distances greater than sum of covalent radii plus
                distance_cutoffs[0]. Distances greater than covalent radii sum
                plus distance_cutoffs[1] are enforced to have zero weight.
            x_diff_weight: (float) - if multiple types of neighbor elements are
                possible, this sets preferences for targets with higher
                electronegativity difference.
            porous_adjustment: (bool) - if True, readjusts Voronoi weights to
                better describe layered / porous structures
            search_cutoff: (float) cutoff in Angstroms for initial neighbor
                search; this will be adjusted if needed internally
            fingerprint_length: (int) if a fixed_length CN "fingerprint" is
                desired from get_nn_data(), set this parameter
        """
        self.weighted_cn = weighted_cn
        self.cation_anion = cation_anion
        self.distance_cutoffs = distance_cutoffs
        self.x_diff_weight = x_diff_weight if x_diff_weight is not None else 0
        self.search_cutoff = search_cutoff
        self.porous_adjustment = porous_adjustment
        self.fingerprint_length = fingerprint_length

    @property
    def structures_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return False

    def get_nn_info(self, structure, n):
        """
        Get all near-neighbor information.
        Args:
            structure: (Structure) pymatgen Structure
            n: (int) index of target site
        Returns:
            siw (list of dicts): each dictionary provides information
                about a single near neighbor, where key 'site' gives
                access to the corresponding Site object, 'image' gives
                the image location, and 'weight' provides the weight
                that a given near-neighbor site contributes
                to the coordination number (1 or smaller), 'site_index'
                gives index of the corresponding site in
                the original structure.
        """

        nndata = self.get_nn_data(structure, n)

        if not self.weighted_cn:
            max_key = max(nndata.cn_weights, key=lambda k: nndata.cn_weights[k])
            nn = nndata.cn_nninfo[max_key]
            for entry in nn:
                entry['weight'] = 1
            return nn

        else:
            for entry in nndata.all_nninfo:
                weight = 0
                for cn in nndata.cn_nninfo:
                    for cn_entry in nndata.cn_nninfo[cn]:
                        if entry['site'] == cn_entry['site']:
                            weight += nndata.cn_weights[cn]

                entry['weight'] = weight

            return nndata.all_nninfo

    def get_nn_data(self, structure, n, length=None):
        """
        The main logic of the method to compute near neighbor.
        Args:
            structure: (Structure) enclosing structure object
            n: (int) index of target site to get NN info for
            length: (int) if set, will return a fixed range of CN numbers
        Returns:
            a namedtuple (NNData) object that contains:
                - all near neighbor sites with weights
                - a dict of CN -> weight
                - a dict of CN -> associated near neighbor sites
        """

        length = length or self.fingerprint_length

        # determine possible bond targets
        target = None
        if self.cation_anion:
            target = []
            m_oxi = structure[n].specie.oxi_state
            for site in structure:
                if site.specie.oxi_state * m_oxi <= 0:  # opposite charge
                    target.append(site.specie)
            if not target:
                raise ValueError('No valid targets for site within cation_anion constraint!')

        # get base VoronoiNN targets
        cutoff = self.search_cutoff
        vnn = VoronoiNN(
            weight='solid_angle',
            targets=target,
            cutoff=cutoff,
            compute_adj_neighbors=False,
        )
        nn = vnn.get_nn_info(structure, n)

        # solid angle weights can be misleading in open / porous structures
        # adjust weights to correct for this behavior
        if self.porous_adjustment:
            for x in nn:
                x['weight'] *= x['poly_info']['solid_angle'] / x['poly_info']['area']

        # adjust solid angle weight based on electronegativity difference
        if self.x_diff_weight > 0:
            for entry in nn:
                X1 = structure[n].specie.X
                X2 = entry['site'].specie.X

                if math.isnan(X1) or math.isnan(X2):
                    chemical_weight = 1
                else:
                    # note: 3.3 is max deltaX between 2 elements
                    chemical_weight = 1 + self.x_diff_weight * math.sqrt(abs(X1 - X2) / 3.3)

                entry['weight'] = entry['weight'] * chemical_weight

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x['weight'], reverse=True)
        if nn[0]['weight'] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        # renormalize weights so the highest weight is 1.0
        highest_weight = nn[0]['weight']
        for entry in nn:
            entry['weight'] = entry['weight'] / highest_weight

        # adjust solid angle weights based on distance
        if self.distance_cutoffs:
            r1 = _get_radius(structure[n])
            for entry in nn:
                r2 = _get_radius(entry['site'])
                if r1 > 0 and r2 > 0:
                    d = r1 + r2
                else:
                    d = _get_default_radius(structure[n]) + _get_default_radius(entry['site'])

                dist = np.linalg.norm(structure[n].coords - entry['site'].coords)

                _adjust_solid_angle_weight(entry, self.distance_cutoffs, dist, d)

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x['weight'], reverse=True)
        if nn[0]['weight'] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        for entry in nn:
            entry['weight'] = round(entry['weight'], 3)
            del entry['poly_info']  # trim

        # remove entries with no weight
        nn = [x for x in nn if x['weight'] > 0]

        # get the transition distances, i.e. all distinct weights
        dist_bins = []
        for entry in nn:
            if not dist_bins or dist_bins[-1] != entry['weight']:
                dist_bins.append(entry['weight'])
        dist_bins.append(0)

        # main algorithm to determine fingerprint from bond weights
        cn_weights = {}  # CN -> score for that CN
        cn_nninfo = {}  # CN -> list of nearneighbor info for that CN
        for idx, val in enumerate(dist_bins):
            if val != 0:
                nn_info = []
                for entry in nn:
                    if entry['weight'] >= val:
                        nn_info.append(entry)
                cn = len(nn_info)
                cn_nninfo[cn] = nn_info
                cn_weights[cn] = _semicircle_integral(dist_bins, idx)

        # add zero coord
        cn0_weight = 1.0 - sum(cn_weights.values())
        if cn0_weight > 0:
            cn_nninfo[0] = []
            cn_weights[0] = cn0_weight

        return self.transform_to_length(self.NNData(nn, cn_weights, cn_nninfo), length)

    def get_cn(self, structure, n, use_weights=False):
        """
        Get coordination number, CN, of site with index n in structure.
        Args:
            structure (Structure): input structure.
            n (integer): index of site for which to determine CN.
            use_weights (boolean): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).
        Returns:
            cn (integer or float): coordination number.
        """
        if self.weighted_cn != use_weights:
            raise ValueError('The weighted_cn parameter and use_weights ' 'parameter should match!')

        return super().get_cn(structure, n, use_weights)

    def get_cn_dict(self, structure, n, use_weights=False):
        """
        Get coordination number, CN, of each element bonded to site with index n in structure
        Args:
            structure (Structure): input structure
            n (integer): index of site for which to determine CN.
            use_weights (boolean): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).
        Returns:
            cn (dict): dictionary of CN of each element bonded to site
        """
        if self.weighted_cn != use_weights:
            raise ValueError('The weighted_cn parameter and use_weights ' 'parameter should match!')

        return super().get_cn_dict(structure, n, use_weights)

    @staticmethod
    def transform_to_length(nndata, length):
        """
        Given NNData, transforms data to the specified fingerprint length
        Args:
            nndata: (NNData)
            length: (int) desired length of NNData
        """

        if length is None:
            return nndata

        if length:
            for cn in range(length):
                if cn not in nndata.cn_weights:
                    nndata.cn_weights[cn] = 0
                    nndata.cn_nninfo[cn] = []

        return nndata


@jit(nopython=True)
def _semicircle_integral(dist_bins, idx):
    """
    An internal method to get an integral between two bounds of a unit
    semicircle. Used in algorithm to determine bond probabilities.
    Args:
        dist_bins: (float) list of all possible bond weights
        idx: (float) index of starting bond weight
    Returns:
        (float) integral of portion of unit semicircle
    """
    r = 1

    x1 = dist_bins[idx]
    x2 = dist_bins[idx + 1]

    if dist_bins[idx] == 1:
        area1 = 0.25 * math.pi * r**2
    else:
        area1 = 0.5 * ((x1 * math.sqrt(r**2 - x1**2)) + (r**2 * math.atan(x1 / math.sqrt(r**2 - x1**2))))

    area2 = 0.5 * ((x2 * math.sqrt(r**2 - x2**2)) + (r**2 * math.atan(x2 / math.sqrt(r**2 - x2**2))))

    return (area1 - area2) / (0.25 * math.pi * r**2)


@jit(nopython=True)
def _adjust_solid_angle_weight(entry, distance_cutoffs, dist, d):
    cutoff_low = d + distance_cutoffs[0]
    cutoff_high = d + distance_cutoffs[1]

    if dist <= cutoff_low:
        dist_weight = 1
    elif dist < cutoff_high:
        dist_weight = (math.cos((dist - cutoff_low) / (cutoff_high - cutoff_low) * math.pi) + 1) * 0.5
    entry['weight'] = entry['weight'] * dist_weight
