# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Run the featurization on the structures from Materials Project
"""

import concurrent.futures
import os
from glob import glob

from tqdm import tqdm

from oximachine_featurizer.featurize import GetFeatures

MPDIR = "/Users/kevinmaikjablonka/Dropbox (LSMO)/proj62_guess_oxidation_states/mp_structures"
ALREADY_FEATURIZED = glob(os.path.join(MPDIR, "*.pkl"))
OUTDIR = (
    "/Users/kevinmaikjablonka/Dropbox (LSMO)/proj62_guess_oxidation_states//mp_features"
)


def featurize_single(structure, outdir=OUTDIR):
    """Featurize one structure"""
    gf = GetFeatures.from_file(structure, outdir)  # pylint:disable=invalid-name
    gf._run_featurization()  # pylint:disable=protected-access


def main():
    """CLI"""
    all_structures = glob(os.path.join(MPDIR, "*.cif"))
    print(f"found {len(all_structures)} structures")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for _ in tqdm(
            executor.map(featurize_single, all_structures), total=len(all_structures)
        ):
            pass


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
