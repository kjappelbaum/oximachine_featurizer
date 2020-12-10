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


Feauturizing a structure
-------------------------



Additional tools
------------------

Scripts that are prefixed with an underscore are part of the private API and may contain hard coded paths. For example, :code:`_run_featurization_slurm_serial.py` contains code that is specific to our cluster infrastructure. 