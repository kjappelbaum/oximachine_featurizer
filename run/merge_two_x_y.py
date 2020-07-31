# -*- coding: utf-8 -*-
"""
Merge two X/y. Takes care of reading two feature/label files, merges them and shuffels them
and writes them to new output directory.

Of course, I could also generalize to two lists of X and ys, respectively but this might lead
to even worse practices.

One big approximation of all of this is that all data fits into memory in once. But it should not be
hard to wrap it in Dask if this is not the case.
"""

import os
import pickle

import click
import numpy as np
from sklearn.utils import shuffle

from oximachine_featurizer.utils import read_pickle

RANDOM_SEED = 1234


class Merger:
    """Class to merge two featrue sets"""

    def __init__(  # pylint:disable=too-many-arguments
        self,
        features0,
        features1,
        labels0,
        labels1,
        names0,
        names1,
        outdir_features,
        outdir_labels,
        outdir_names,
    ):

        self.features0 = features0
        self.features1 = features1
        # make sure that they have the same number of features
        assert self.features0.shape[1] == self.features1.shape[1]
        self.labels0 = labels0
        self.labels1 = labels1

        self.names0 = names0
        self.names1 = names1
        # make sure labels have the same number of columns (one) and the same length as the corresponding features
        assert len(self.features0) == len(self.labels0)
        assert len(self.features1) == len(self.labels1)
        assert len(self.labels0) == len(self.names0)
        assert len(self.labels1) == len(self.labels1)

        # set the outdir
        self.outdir_features = outdir_features
        self.outdir_labels = outdir_labels
        self.outdir_names = outdir_names

    @staticmethod
    def stack_arrays(features0, features1, labels0, labels1, names0, names1):
        """Perform the actual merging"""
        X = np.vstack([features0, features1])  # pylint:disable=invalid-name
        y = np.array(list(labels0) + list(labels1))  # pylint:disable=invalid-name
        names = names0 + names1
        return X, y, names

    @classmethod
    def from_files(  # pylint:disable=too-many-arguments
        cls,
        features0path,
        features1path,
        labels0path,
        labels1path,
        names0path,
        names1path,
        outdir_features,
        outdir_labels,
        outdir_names,
    ):
        """Construct class from filepaths"""
        features0 = np.load(features0path)
        features1 = np.load(features1path)
        labels0 = np.load(labels0path)
        labels1 = np.load(labels1path)
        names0 = read_pickle(names0path)
        names1 = read_pickle(names1path)

        return cls(
            features0,
            features1,
            labels0,
            labels1,
            names0,
            names1,
            outdir_features,
            outdir_labels,
            outdir_names,
        )

    @staticmethod
    def output(  # pylint:disable = invalid-name
        X,  # pylint:disable = invalid-name
        y,  # pylint:disable = invalid-name
        names,
        outdir_features,
        outdir_labels,
        outdir_names,
    ):
        """Write the new training set files for the merged training set"""
        features, labels, names = shuffle(X, y, names, random_state=RANDOM_SEED)

        np.save(os.path.join(outdir_features, 'features'), features)
        np.save(os.path.join(outdir_labels, 'labels'), labels)

        with open(os.path.join(outdir_names, 'names.pkl'), 'wb') as picklefile:
            pickle.dump(names, picklefile)

    def merge(self):
        """Stack arrays and shuffle"""
        X, y, names = Merger.stack_arrays(  # pylint:disable=invalid-name
            self.features0,
            self.features1,
            self.labels0,
            self.labels1,
            self.names0,
            self.names1,
        )

        # Now shuffle and output
        Merger.output(X, y, names, self.outdir_features, self.outdir_labels, self.outdir_names)


@click.command('cli')
@click.argument('features0path')
@click.argument('features1path')
@click.argument('labels0path')
@click.argument('labels1path')
@click.argument('names0path')
@click.argument('names1path')
@click.argument('outdir_features')
@click.argument('outdir_labels')
@click.argument('outdir_names')
def run_merging(  # pylint:disable=too-many-arguments
    features0path,
    features1path,
    labels0path,
    labels1path,
    names0path,
    names1path,
    outdir_features,
    outdir_labels,
    outdir_names,
):
    """CLI"""
    merger = Merger.from_files(
        features0path,
        features1path,
        labels0path,
        labels1path,
        names0path,
        names1path,
        outdir_features,
        outdir_labels,
        outdir_names,
    )
    merger.merge()


if __name__ == '__main__':
    run_merging()  # pylint:disable=no-value-for-parameter
