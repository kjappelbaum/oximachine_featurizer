# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Status: Dev
Run the featurization on one structure
"""
from __future__ import absolute_import
import click
from mine_mof_oxstate.featurize import FeatureCollector


@click.command('cli')
@click.argument('inpath')
@click.argument('labelpath')
@click.argument('outdir_labels')
@click.argument('outdir_features')
@click.argument('outdir_helper')
def main(inpath, labelpath, outdir_labels, outdir_features, outdir_helper):
    """
    CLI function
    """

    fc = FeatureCollector(inpath, labelpath, outdir_labels, outdir_features, outdir_helper)  # pylint:disable=invalid-name
    fc.dump_featurecollection()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
