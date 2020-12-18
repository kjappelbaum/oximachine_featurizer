# -*- coding: utf-8 -*-
"""Run the featurization on one structure"""

import click
import numpy as np
from pymatgen import Structure

from oximachine_featurizer import featurize


@click.command("cli")
@click.argument("structure")
@click.argument("outname")
def main(structure: str, outname: str):
    """CLI function"""
    structure = Structure.from_file(structure)
    feature_matrix, _, _ = featurize(structure)

    np.save(outname, feature_matrix)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
