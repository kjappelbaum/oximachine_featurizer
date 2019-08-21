# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Status: Dev
Run the featurization on the CSD MOF subset
"""
from __future__ import absolute_import
import os
import concurrent.futures
from glob import glob
from pathlib import Path
from tqdm import tqdm
from mine_mof_oxstate.featurize import GetFeatures

OUTDIR = '/home/kevin/Dropbox/proj62_guess_oxidation_states/mine_csd/featurization'
CSDDIR = '/home/kevin/lsmo_db_share/shared/db_structures/mof_subset_csdmay2019'
ALREADY_FEAUTRIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, '*.pkl'))]


def featurize_single(structure, outdir=OUTDIR):
    if Path(structure).stem not in ALREADY_FEAUTRIZED:
        gf = GetFeatures(structure, outdir)  # pylint:disable=invalid-name
        gf.run_featurization()


def main():
    all_structures = glob(os.path.join(CSDDIR, '*.cif'))
    with concurrent.futures.ProcessPoolExecutor as executor:
        for _ in tqdm(executor.map(featurize_single, all_structures), total=len(all_structures)):
            pass


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
