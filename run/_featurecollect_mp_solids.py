# -*- coding: utf-8 -*-
"""
Collect features and labels for the structures from Materials Project
"""
import os
import pickle
import tempfile

import click
import pandas as pd

from oximachine_featurizer.featurize import FeatureCollector


def write_labels_to_stupid_format(df, outdir):  # pylint:disable = invalid-name
    """Write the nice columns from the pd DataFrame we got after
    parsing materialsproject into something that is consistent
    with the automatic featurecollection to keep our lives easier"""
    stupid_dict = {}
    for _, row in df.iterrows():
        stupid_dict[row['name']] = {row['metal']: [row['oxidationstate']]}

    with open(os.path.join(outdir, 'materialsproject_structure_labels.pkl'), 'wb') as fh:  # pylint:disable=invalid-name
        pickle.dump(stupid_dict, fh)


@click.command('cli')
@click.argument('dfpath')
@click.argument('inpath')
@click.argument('outdir_labels')
@click.argument('outdir_features')
@click.argument('outdir_helper')
@click.argument('percentage_holdout')
@click.argument('outdir_holdout')
@click.argument('training_set_size')
@click.argument('features', nargs=-1)
def main(  # pylint:disable=too-many-arguments
    dfpath,
    inpath,
    outdir_labels,
    outdir_features,
    outdir_helper,
    percentage_holdout,
    outdir_holdout,
    training_set_size,
    features,
):
    """CLI"""
    df = pd.read_csv(dfpath)  # pylint:disable=invalid-name
    dirpath = tempfile.mkdtemp()
    write_labels_to_stupid_format(df, outdir=dirpath)

    print(f'rewrote labels to {dirpath}')
    try:
        training_set_size = int(training_set_size)
    except Exception:  # pylint:disable=broad-except
        # if it is None, it will not use farthest point sampling to create a smaller set
        training_set_size = None

    fc = FeatureCollector(  # pylint:disable=invalid-name
        inpath,
        os.path.join(dirpath, 'materialsproject_structure_labels.pkl'),
        outdir_labels,
        outdir_features,
        outdir_helper,
        float(percentage_holdout),
        outdir_holdout,
        training_set_size=training_set_size,
        selected_features=features,
    )
    fc.dump_featurecollection()


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
