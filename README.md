# oximachine_featurizer

[![Actions Status](https://github.com/kjappelbaum/mof_oxidation_states/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/mof_oxidation_states/actions)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3567274.svg)](https://doi.org/10.5281/zenodo.3567274)
[![Maintainability](https://api.codeclimate.com/v1/badges/936cc6cc791f8bf352c6/maintainability)](https://codeclimate.com/github/kjappelbaum/mof_oxidation_states/maintainability)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kjappelbaum/oximachine_featurizer/master?filepath=examples%2Fexample.ipynb)

Mine oxidation states for structures from the (MOF) subset of the CSD and calculate features for them. Runscripts are automatically installed for the most important steps. Some of these runscripts contain hardcoded paths, that would need to be updated.
This code generates inputs that can be used with the [learnmofox package](https://github.com/kjappelbaum/learn_mof_ox_state.git).

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

- To run the featurization

```bash
run_featurization {structure} {outdir}
```

for each metal center this should take seconds if there is no disorder.
Note that the metal center features are added using methods from the `FeatureCollector` class.

- To collect separate files with features into one file for the feature matrix, you can use the featurecollector, e.g.

```bash
run_featurecollection --only_racs {FEATURESPATH}  {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000 {RACSDATAPATH} column row crystal_nn_no_steinhardt
```

The bottleneck of this approach is that it currently checks for each name if we exclude it (e.g., due to wrong assignments). One should expect a runtime in the order of several minutes for several structures.

Some output can be found on the [MaterialsCloud Archive (doi: 10.24435/materialscloud:2019.0085/v1 )](https://doi.org/10.24435/materialscloud:2019.0085/v1).

## Example usage

The use of the main functions of this package is shown in the Jupyter Notebook in the example directory.
It contains some example structures and the output, which should be produces in seconds.

## Testing the installation

For testing, you can---as it is done for the continuous integration (CI)---use `pytest` and run the files in the `test` directory. For example

```(bash)
pip install pytest
pytest test/test_featurize.py
```
