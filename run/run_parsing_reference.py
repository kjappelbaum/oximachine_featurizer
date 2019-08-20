# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Status: Dev
Run the oxidation state mining
"""
from __future__ import absolute_import
import time
import pickle
import click
from ccdc import io  # pylint: disable=import-error
from mine_mof_oxstate.parse import GetOxStatesCSD


def run_parsing(output_name=None):
    """

    Args:
        output_name (str): filestem for the output pickle file

    Returns:
        writes output as pickle file

    """
    csd_reader = io.EntryReader('CSD')  # all database entries
    getoxstatesobject = GetOxStatesCSD(csd_reader)
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
