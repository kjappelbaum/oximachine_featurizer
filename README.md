# mine_mof_oxstate

[![Actions Status](https://github.com/kjappelbaum/mof_oxidation_states/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/mof_oxidation_states/actions)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3567274.svg)](https://doi.org/10.5281/zenodo.3567274)
[![Maintainability](https://api.codeclimate.com/v1/badges/936cc6cc791f8bf352c6/maintainability)](https://codeclimate.com/github/kjappelbaum/mof_oxidation_states/maintainability)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Mine oxidations states for structures from the (MOF) subset of the CSD and calculate features for them. Runscripts are automatically installed for the most important steps.
This code generates inputs that can be used with the [learnmofox package](https://github.com/kjappelbaum/learn_mof_ox_state.git).

> ⚠️ **Warning**: Some parts of the code require some modifications in the dependencies, for which I did not make PRs so far. You need to use my forks. You need `pip>=18.1` for this to be set up automatically. More details can be found below.

> ⚠️ **Warning**: For the mining of the oxidation states, you need the CSD Python API.
> You need to export the `CSD_HOME` path. Due to the licensing issues, this cannot be done automatically.

## Installation

To install the software with all dependencies, you can use

```bash
pip install git+https://github.com/kjappelbaum/mof_oxidation_states.git
```

This should, for appropriate versions of pip (`pip>=18.1`), also install [our fork of matminer from the correct branch](https://github.com/kjappelbaum/matminer.git@localpropertystats).
This automatically installs several command-line tools (CLI) which are detailed below.

The full process should take some seconds.

## How to use it

- To run the featurization

```bash
run_featurization {structure} {outdir}
```

for each metal center this should take seconds if there is no disorder.

- To collect separate files with features into one file for the feature matrix, you can use the featurecollector, e.g.

```bash
run_featurecollection --only_racs {FEATURESPATH}  {LABELSPATH} {labelsoutpath} {featureoutspath} {helperoutpath} 0.2 {holdoutpath} 60000 {RACSDATAPATH} column row crystal_nn_no_steinhardt'
```

The bottleneck of this approach is that it currently checks for each name if we exclude it (e.g., due to wrong assignments). One should expect a runtime in the order of several minutes for several structures.

Some output can be found on the [MaterialsCloud Archive (doi: 10.24435/materialscloud:2019.0085/v1 )](https://doi.org/10.24435/materialscloud:2019.0085/v1).
