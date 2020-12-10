# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level, line-too-long, too-many-arguments
"""
Status: Dev
Run the featurization on one structure
"""

import os

import click

from oximachine_featurizer.featurize import FeatureCollector


@click.command('cli')
@click.argument('inpath', type=click.Path(exists=True))
@click.argument('labelpath', type=click.Path(exists=True))
@click.argument('outdir_labels', type=click.Path(exists=True))
@click.argument('outdir_features', type=click.Path(exists=True))
@click.argument('outdir_helper', type=click.Path(exists=True))
@click.argument('percentage_holdout')
@click.argument('outdir_holdout', type=click.Path(exists=True))
@click.argument('training_set_size')
@click.argument('racsfile')
@click.argument('features', nargs=-1)
@click.option('--only_racs', is_flag=True)
@click.option('--do_not_drop_duplicates', is_flag=True)
def main(
    inpath,
    labelpath,
    outdir_labels,
    outdir_features,
    outdir_helper,
    percentage_holdout,
    outdir_holdout,
    training_set_size,
    racsfile,
    features,
    only_racs,
    do_not_drop_duplicates,
):
    """
    CLI function
    """

    do_not_drop_duplicates = not do_not_drop_duplicates

    try:
        training_set_size = int(training_set_size)
    except Exception:  # pylint:disable=broad-except
        # if it is None, it will not use farthest point sampling to create a smaller set
        training_set_size = None

    if not os.path.exists(racsfile):
        racsfile = None

    if only_racs:
        features = []

    fc = FeatureCollector(  # pylint:disable=invalid-name
        inpath,
        labelpath,
        outdir_labels,
        outdir_features,
        outdir_helper,
        float(percentage_holdout),
        outdir_holdout,
        training_set_size=training_set_size,
        racsfile=racsfile,
        selected_features=features,
        drop_duplicates=do_not_drop_duplicates,
    )
    fc.dump_featurecollection()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
