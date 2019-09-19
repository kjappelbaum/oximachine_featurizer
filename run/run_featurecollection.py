# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level, line-too-long, too-many-arguments
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
@click.argument('percentage_holdout')
@click.argument('outdir_holdout')
@click.argument('features', nargs=-1)
def main(
        inpath,
        labelpath,
        outdir_labels,
        outdir_features,
        outdir_helper,
        percentage_holdout,
        outdir_holdout,
        features,
):
    """
    CLI function
    """

    fc = FeatureCollector(  # pylint:disable=invalid-name
        inpath,
        labelpath,
        outdir_labels,
        outdir_features,
        outdir_helper,
        float(percentage_holdout),
        outdir_holdout,
        selected_features=features,
    )
    fc.dump_featurecollection()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
