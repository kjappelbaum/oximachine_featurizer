# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Run the featurization on the structures from Materials Project
"""

import concurrent.futures
import os
import pickle
from glob import glob

from tqdm import tqdm

from oximachine_featurizer.featurize import GetFeatures

MPDIR = '/Users/kevinmaikjablonka/Dropbox (LSMO)/proj62_guess_oxidation_states/mp_structures'
ALREADY_FEATURIZED = glob(os.path.join(MPDIR, '*.pkl'))
OUTDIR = ('/Users/kevinmaikjablonka/Dropbox (LSMO)/proj62_guess_oxidation_states//mp_features')


def load_pickle(f):  # pylint:disable=invalid-name
    with open(f, 'rb') as fh:  # pylint:disable=invalid-name
        result = pickle.load(fh)
    return result


def featurize_single(structure, outdir=OUTDIR):
    gf = GetFeatures.from_file(structure, outdir)  # pylint:disable=invalid-name
    gf.run_featurization()


def main():
    """CLI"""
    all_structures = glob(os.path.join(MPDIR, '*.cif'))
    print(f'found {len(all_structures)} structures')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for _ in tqdm(executor.map(featurize_single, all_structures), total=len(all_structures)):
            pass


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
