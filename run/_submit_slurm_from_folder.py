# -*- coding: utf-8 -*-
# pylint:disable = logging-format-interpolation
"""
This runscript can be used to submit the featurizations on a HPC clusters with the SLURM workload manager.

Usage: Install script in conda enviornment called ml on cluster and then run it using the
the outdir, start and end indices and submit flag.
"""

import logging
import os
import subprocess
import time
from glob import glob
from pathlib import Path

import click

featurizer = logging.getLogger('featurizer')  # pylint:disable=invalid-name
featurizer.setLevel(logging.DEBUG)
logging.basicConfig(filename='featurizer.log', format='%(filename)s: %(message)s', level=logging.DEBUG)

THIS_DIR = os.path.dirname(__file__)

OUTDIR = '/scratch/kjablonk/oximachine_all'
INDIR = '/work/lsmo/jablonka/2020-4-7_all_csd_for_oximachine/cif_for_feat'
ALREADY_FEAUTRIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, '*.pkl'))]

SUBMISSION_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       5GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task    1
#SBATCH --job-name  {name}
#SBATCH --time      0:10:00
#SBATCH --partition=serial

source /home/kjablonk/anaconda3/bin/activate
conda activate ml

run_featurization {structure} {outdir}
"""

all_structures = sorted(glob(os.path.join(INDIR, '*.cif')))


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
    for structure in all_structures[start:end:5]:
        if structure not in ALREADY_FEAUTRIZED:
            name = Path(structure).stem
            write_and_submit_slurm(THIS_DIR, name, structure, outdir, submit)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
