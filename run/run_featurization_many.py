# -*- coding: utf-8 -*-
"""
Status: Dev
Run the featurization on the CSD MOF subset
"""
import concurrent.futures
import os
import pickle
from glob import glob
from pathlib import Path

import click
from tqdm import tqdm

from oximachine_featurizer.featurize import GetFeatures

OUTDIR = '/scratch/kjablonk/oximachine_all/features'
INDIR = '/work/lsmo/jablonka/2020-4-7_all_csd_for_oximachine/cif_for_feat'
ALREADY_FEATURIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, '*.pkl'))]


def read_already_featurized():
    if os.path.exists('already_featurized.txt'):
        with open('already_featurized.txt', 'r') as fh:
            already_featurized = fh.readlines()
        ALREADY_FEATURIZED.extend(already_featurized)


def load_pickle(f):  # pylint:disable=invalid-name
    with open(f, 'rb') as fh:  # pylint:disable=invalid-name
        result = pickle.load(fh)
    return result


def featurize_single(structure, outdir=OUTDIR):
    if Path(structure).stem not in ALREADY_FEATURIZED:
        try:
            gf = GetFeatures.from_file(structure, outdir)  # pylint:disable=invalid-name
            gf.run_featurization()
        except Exception:
            pass


@click.command('cli')
@click.option('--reverse', is_flag=True)
def main(reverse):
    read_already_featurized()
    if reverse:
        all_structures = sorted(glob(os.path.join(INDIR, '*.cif')), reverse=True)
    else:
        all_structures = sorted(glob(os.path.join(INDIR, '*.cif')))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in tqdm(
                list(executor.map(featurize_single, all_structures)),
                total=len(all_structures),
        ):
            pass


if __name__ == '__main__':
    print(('working in {}'.format(INDIR)))
    main()  # pylint: disable=no-value-for-parameter
