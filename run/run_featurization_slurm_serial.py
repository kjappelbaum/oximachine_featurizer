# -*- coding: utf-8 -*-
# pylint:disable = logging-format-interpolation
"""
Status: Dev
This runscript can be used to submit the featurizations on a HPC clusters with the SLURM workload manager.

Usage: Install script in conda enviornment called ml on cluster and then run it using the
the outdir, start and end indices and submit flag.
"""

import logging
import os
import pickle
import subprocess
import time
from glob import glob
from pathlib import Path

import click

featurizer = logging.getLogger('featurizer')  # pylint:disable=invalid-name
featurizer.setLevel(logging.DEBUG)
logging.basicConfig(filename='featurizer.log', format='%(filename)s: %(message)s', level=logging.DEBUG)

THIS_DIR = os.path.dirname(__file__)

OUTDIR = '/scratch/kjablonk/proj62_featurization/extended_chemspace'
CSDDIR = '/work/lsmo/mof_subset_csdmay2019'
ALREADY_FEAUTRIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, '*.pkl'))]
NAME_LIST = '/scratch/kjablonk/oxidationstates/to_sample_new.pkl'

SUBMISSION_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       5GB
#SBATCH --ntasks    1
#SBATCH --job-name  {name}
#SBATCH --time      1:00:00
#SBATCH --partition=serial

source /home/kjablonk/anaconda3/bin/activate
conda activate ml

run_featurization {structure} {outdir}
"""


def load_pickle(f):  # pylint:disable=invalid-name
    """Loads a pickle file"""
    with open(f, 'rb') as fh:  # pylint:disable=invalid-name
        result = pickle.load(fh)
    return result


TO_SAMPLE = load_pickle(NAME_LIST)


def write_and_submit_slurm(workdir, name, structure, outdir, submit=False):
    """writes a slurm submission script and submits it if requested"""
    submission_template = SUBMISSION_TEMPLATE.format(name=name + '_featurize', structure=structure, outdir=outdir)
    with open(os.path.join(workdir, name + '.slurm'), 'w') as fh:  # pylint:disable=invalid-name
        for line in submission_template:
            fh.write(line)

    featurizer.info('prepared {} for submission'.format(name))
    if submit:
        subprocess.call('sbatch {}'.format('{}.slurm'.format(name)), shell=True)
        time.sleep(2)
        featurizer.info('submitted {}'.format(name))


@click.command('cli')
@click.argument('outdir')
@click.argument('start')
@click.argument('end')
@click.option('--submit', is_flag=True, help='actually submit slurm job')
def main(outdir, start, end, submit):
    """Runs the CLI"""
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    start = int(start)
    end = int(end)
    for structure in TO_SAMPLE[start:end:5]:
        if structure not in ALREADY_FEAUTRIZED:
            name = structure
            write_and_submit_slurm(THIS_DIR, name, os.path.join(CSDDIR, structure + '.cif'), outdir, submit)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
