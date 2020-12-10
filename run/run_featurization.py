# -*- coding: utf-8 -*-
# pylint:disable=relative-beyond-top-level
"""
Run the featurization on one structure
"""

import click
import numpy as np
from pymatgen import Structure

from oximachine_featurizer import featurize


@click.command("cli")
@click.argument("structure")
@click.argument("outname")
def main(structure, outname):
    """
    CLI function
    """
    structure = Structure.from_file(structure)
    X, _, _ = featurize(structure)  # pylint: disable=invalid-name

    np.save(outname, X)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
