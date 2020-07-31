# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Run the oxidation state mining
"""

import pickle
import random
import time

import click
from ccdc import io  # pylint: disable=import-error

from oximachine_featurizer.parse import GetOxStatesCSD


def generate_id_list(num_samples=1009141):
    """Sample some random entries from the CSD"""
    ids = []
    csd_reader = io.EntryReader('CSD')
    idxs = random.sample(list(range(len(csd_reader))), num_samples)
    for idx in idxs:
        ids.append(csd_reader[idx].identifier)
    return ids


def run_parsing(output_name=None):
    """

    Args:
        output_name (str): filestem for the output pickle file

    Returns:
        writes output as pickle file

    """
    # all database entries
    getoxstatesobject = GetOxStatesCSD(generate_id_list())
    if output_name is None:
        timestr = time.strftime('%Y%m%d-%H%M%S')
        output_name = '-'.join([timestr, 'csd_ox_parse_output_reference'])

    outputdict = getoxstatesobject.run_parsing(njobs=4)

    with open(output_name + '.pkl', 'wb') as filehandle:
        pickle.dump(outputdict, filehandle)


@click.command('cli')
@click.option('--outname', default=None)
def main(outname):
    """
    CLI function
    """
    run_parsing(outname)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
