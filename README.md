# oximachine_featurizer

[![Actions Status](https://github.com/kjappelbaum/mof_oxidation_states/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/mof_oxidation_states/actions)
[![Documentation Status](https://readthedocs.org/projects/oximachine-featurizer/badge/?version=latest)](https://oximachine-featurizer.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3567274.svg)](https://doi.org/10.5281/zenodo.3567274)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kjappelbaum/oximachine_featurizer/master?filepath=examples%2Fexample.ipynb)

Mine oxidation states for structures from the (MOF) subset of the CSD and calculate features for them. Runscripts are automatically installed for the most important steps. Some of these runscripts contain hard coded paths, that would need to be updated.
This code generates inputs that can be used with the [learnmofox package](https://github.com/kjappelbaum/learn_mof_ox_state.git) to replicate our work [1].

If you're just interested in using a pre-trained model, the [oximachinerunner](https://github.com/kjappelbaum/oximachinerunner) package.

> ⚠️ **Warning**: For the mining of the oxidation states, you need the CSD Python API.
> You need to export the `CSD_HOME` path. Due to the licensing issues, this cannot be done automatically.

## Installation

The commands below automatically install several command-line tools (CLI) which are detailed below.

The full process should take some seconds.

### Latest version

To install the latest version of the software with all dependencies, you can use

```bash
pip install git+https://github.com/kjappelbaum/oximachine_featurizer.git
```

### Stable release

```bash
pip install oximachine_featurizer
```

## How to use it

To run the default featurization on one structure you can use the CLI

```bash
run_featurization <structure> <outdir>
```

for each metal center this should take seconds if there is no disorder.

Some output can be found on the [MaterialsCloud Archive (doi: 10.24435/materialscloud:2019.0085/v1 )](https://doi.org/10.24435/materialscloud:2019.0085/v1).

More details can be found in the [documentation](https://oximachine-featurizer.readthedocs.io/en/latest/).

## Example usage

The use of the main functions of this package is shown in the Jupyter Notebook in the example directory.
It contains some example structures and the output, which should be produces in seconds.

## Testing the installation

For testing, you can---as it is done for the continuous integration (CI)---use `pytest` and run the files in the `test` directory. For example

```(bash)
pip install pytest
pytest test/main
```


## References

[1] Jablonka, Kevin Maik; Ongari, Daniele; Moosavi, Seyed Mohamad; Smit, Berend (2020): Using Collective Knowledge to Assign Oxidation States. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.11604129.v1
