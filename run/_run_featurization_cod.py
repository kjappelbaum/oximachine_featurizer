# -*- coding: utf-8 -*-
"""
Status: Dev
Run the featurization on the CSD MOF subset
"""
import concurrent.futures
import os
from glob import glob
from pathlib import Path

import click
from tqdm import tqdm

from oximachine_featurizer.featurize import GetFeatures

OUTDIR = "/scratch/kjablonk/oximachine_all/features_cod"
INDIR = "/work/lsmo/jablonka/cod_to_featurize"
ALREADY_FEATURIZED = [Path(p).stem for p in glob(os.path.join(OUTDIR, "*.pkl"))]


def read_already_featurized():
    """Reads a file with list of already featurized files"""
    if os.path.exists("already_featurized.txt"):
        with open("already_featurized.txt", "r") as handle:
            already_featurized = handle.readlines()
        ALREADY_FEATURIZED.extend(already_featurized)


def featurize_single(structure, outdir=OUTDIR):
    """Runs featurization on one structure."""
    if Path(structure).stem not in ALREADY_FEATURIZED:
        try:
            gf = GetFeatures.from_file(structure, outdir)  # pylint:disable=invalid-name
            gf._run_featurization()  # pylint:disable=protected-access
        except Exception:  # pylint:disable=broad-except
            pass


@click.command("cli")
@click.option("--reverse", is_flag=True)
def main(reverse):
    """CLI"""
    read_already_featurized()
    if reverse:
        all_structures = sorted(glob(os.path.join(INDIR, "*.cif")), reverse=True)
    else:
        all_structures = sorted(glob(os.path.join(INDIR, "*.cif")))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in tqdm(
            list(executor.map(featurize_single, all_structures)),
            total=len(all_structures),
        ):
            pass


if __name__ == "__main__":
    print(("working in {}".format(INDIR)))
    main()  # pylint: disable=no-value-for-parameter
