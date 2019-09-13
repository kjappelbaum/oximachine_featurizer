# -*- coding: utf-8 -*-
"""
development version to convert features from a folder of pickle files
to three pickle files for feature matrix, label vector and names list.

Latter is important to investigate the failures manually
"""

from __future__ import absolute_import
import click


@click.command('cli')
@click.argument('inpath')
@click.argument('labelsfile')
@click.argument('outpath')
def main(inpath, labelsfile, outpath):  #pylint:disable=unused-argument
    """Run the CLI"""
    ...


if __name__ == '__main__':
    main()  #pylint:disable=no-value-for-parameter
