# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Status: Dev
Run the featurization on one structure
"""
from __future__ import absolute_import
import click
from mine_mof_oxstate.featurize import GetFeatures


@click.command('cli')
@click.argument('structure')
@click.argument('outdir')
def main(structure, outdir):
    """
    CLI function
    """
    gf = GetFeatures.from_file(structure, outdir, 60)  # pylint: disable=invalid-name
    gf.run_featurization()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
