# -*- coding: utf-8 -*-
"""
Status: Dev
This runscript can be used to submit the featurizations on a HPC clusters with the SLURM workload manager.

Usage: Install script in conda enviornment called mof_oxidation_state_project on cluster and then run it using the
the outdir, start and end indices and submit flag.
"""
from __future__ import absolute_import
import os
import pickle
import time
from glob import glob
from pathlib import Path
import subprocess
import click

THIS_DIR = os.path.dirname(__file__)

OUTDIR = '/scratch/kjablonk/proj62_featurization/20190915_features'
CSDDIR = '/work/lsmo/mof_subset_csdmay2019'
ALREADY_FEAUTRIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, '*.pkl'))]
NAME_LIST = '/home/kevin/Dropbox (LSMO)/proj62_guess_oxidation_states/mine_csd/analysis/name_list.pkl'

SUBMISSION_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       5GB
#SBATCH --job-name  {name}
#SBATCH --time      5:00:00
#SBATCH --partition=serial

source /home/kjablonk/anaconda3/bin/activate
conda activate mof_oxidation_state_project

run_featurization {structure} {outdir}
"""


def load_pickle(f):  # pylint:disable=invalid-name
    """Loads a pickle file"""
    with open(f, 'rb') as fh:  # pylint:disable=invalid-name
        result = pickle.load(fh)
    return result


HAS_OX_NUMER = load_pickle(NAME_LIST)


def write_and_submit_slurm(workdir, name, structure, outdir, submit=False):
    """writes a slurm submission script and submits it if requested"""
    submission_template = SUBMISSION_TEMPLATE.format(name=name + '_featurize', structure=structure, outdir=outdir)
    with open(os.path.join(workdir, name + '.slurm'), 'w') as fh:  # pylint:disable=invalid-name
        for line in submission_template:
            fh.write(line)

    if submit:
        subprocess.call('sbatch {}'.format('{}.slurm'.format(name)), shell=True, cwd=workdir)
        time.sleep(2)


@click.command('cli')
@click.argument('outdir')
@click.argument('start')
@click.argument('end')
@click.option('--submit', is_flag=True, help='actually submit slurm job')
def main(outdir, start, end, submit):
    """Runs the CLI"""
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    structures = glob(os.path.join(CSDDIR, '*.cif'))
    structures.sort()

    for structure in structures[start:end]:
        if Path(structure).stem not in ALREADY_FEAUTRIZED:
            if Path(structure).stem in HAS_OX_NUMER:
                name = Path(structure).stem
                write_and_submit_slurm(THIS_DIR, name, structure, outdir, submit)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
