# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Run the oxidation state mining
"""

import os
import pickle
import time
from glob import glob
from pathlib import Path

import click

from oximachine_featurizer.parse import GetOxStatesCSD


def prepare_list(indir='/mnt/lsmo_databases/mof_subset_csdmay2019'):
    """

    Args:
        indir (str): path to input directory

    Returns:
        list: filestemms

    """
    names = glob(os.path.join(indir, '*.cif'))
    names_cleaned = [Path(n).stem for n in names]
    return names_cleaned


def run_parsing(names_cleaned, output_name=None):
    """

    Args:
        names_cleaned (list): list of CSD identifiers
        output_name (str): filestem for the output pickle file

    Returns:
        writes output as pickle file

    """
    getoxstatesobject = GetOxStatesCSD(names_cleaned)
    if output_name is None:
        timestr = time.strftime('%Y%m%d-%H%M%S')
        output_name = '-'.join([timestr, 'csd_ox_parse_output'])

    outputdict = getoxstatesobject.run_parsing(njobs=4)

    with open(output_name + '.pkl', 'wb') as filehandle:
        pickle.dump(outputdict, filehandle)


@click.command('cli')
@click.argument('indir', default='/mnt/lsmo_databases/mof_subset_csdmay2019')
@click.argument('outname', default=None)
def main(indir, outname):
    """CLI function"""
    names_cleaned = prepare_list(indir)
    run_parsing(names_cleaned, outname)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
