Getting started
====================

Installation
--------------

We recommend installing oximachine_featurizer in a clean virtual environment environment (e.g., a `conda environment <https://docs.conda.io/projects/conda/en/latest/index.html>`_)
The latest stable release can be installed from the Python package index (PyPi):

.. code:: bash

    pip install oximachine_featurizer

The development version can be installed directly from GitHub

.. code:: bash

    pip install git+https://github.com/kjappelbaum/oximachine_featurizer.git


Featurizing a structure
--------------------------

To featurize one structure with the default options you can use the following Python snippet

.. code:: python

    from oximachine_featurizer import featurize
    X, metal_indices, metals = featurize(structure)

Where :code:`structure` is a :code:`pymatgen.Structure` object.
Under the hood, this function calls two different classes, the :py:obj:`~oximachine_featurizer.featurize.GetFeatures` class that computes all features that we considered during development and the :py:obj:`~oximachine_featurizer.featurize.FeatureCollector` that selects the relevant ones.


Alternatively, if you want to featurize directly on the command line, you can use the following syntax

.. code:: bash

    run_featurization <structurefile> <outname>

For example,

.. code:: bash

    run_featurization examples/structures/ACODAA.cif test.npy

This command line tool will attempt to read the :code:`structurefile` using pymatgen and then write the features as ``npy` <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_ file to :code:`outname`. The numpy array in this file can be feed directly into the :code:`StandardScaler` and :code:`VotingClassifier` objects that can be created with the :code:`learnmofox` Python package.


Additional tools
------------------

Scripts that are prefixed with an underscore are part of the private API and may contain hard coded paths. For example, :code:`_run_featurization_slurm_serial.py` contains code that is specific to our cluster infrastructure.


Parsing the CSD
.................

The :py:class:`~oximachine_featurizer.parse.GetOxStatesCSD` can be used to retrieve the oxidation states from a list of CSD identifiers. This feature requires a CSD license and you need to export :code:`CSD_HOME` for the `CSD API <https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html>`_ to work.

You can for example use the following snippet of Python

.. code-block:: python

    from oximachine_featurizer.parse import GetOxStatesCSD
    getoxstates_instance= GetOxStatesCSD(names_cleaned)

    outputdict = getoxstates_instance.run_parsing(njobs=4)

:code:`outputdict` will be a nested dictionary of the form :code:`{'id': {'symbol': [oxidation states]}}`.

Parsing the Materials Project
................................

Using this code requires that you export the :code:`MP_API_KEY` environment variable containing your API key for the Materials Project.
For example, the :py:mod:`oximachine_featurizer.run.run_mine_mp.py` script will retrieve all binary halides, sulfides, oxides, ... that are stable (zero energy above comblex hull) and calculate the oxidation states.
