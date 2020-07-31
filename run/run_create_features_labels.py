# -*- coding: utf-8 -*-
"""
development version to convert features from a folder of pickle files
to three pickle files for feature matrix, label vector and names list.

Latter is important to investigate the failures manually
"""

import click

from oximachine_featurizer.featurize import FeatureCollector


@click.command('cli')
@click.argument('inpath')
@click.argument('labelsfile')
@click.argument('outdir')
def main(inpath, labelsfile, outdir):  #pylint:disable=unused-argument
    """Run the CLI"""
    fc = FeatureCollector(inpath, labelsfile, outdir)  #pylint:disable=invalid-name
    fc.dump_featurecollection()


if __name__ == '__main__':
    main()  #pylint:disable=no-value-for-parameter
