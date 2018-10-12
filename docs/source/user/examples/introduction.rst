Introduction
------------

A collection of examples for running large-eddy simulations of wind plant
aerodynamics is available under :file:`nalu-wind/examples/`.
A :file:`setup.yaml` file is included with all cases.
This file has the most revelant parameters
that can be modified for each case of interest.

A set of Python utilities is included with the examples.
These utilities are meant to simplify the process of generating input files
for running Nalu and plot results form the simulations.

Each case has a default collection of template input files in
:file:`./template_input_files`.
The executable :program:`nalu_input_fileX` will take the input files and
modify them according to the inputs set in :file:`setup.yaml` and
generate new and ready to use input files.

Instructions to compile :program:`Nalu` are provided in :ref:`building_nalu`.
The `wind-utils <https://github.com/Exawind/wind-utils>`_ repository
needs to be compiled to have a access to all the
pre-processing utilities used in the example cases.
The :program:`wind-utils` repository can be downloaded as a submodule by running
this command inside the :file:`nalu-wind/` repository::

    git submodule init && git submodule update

Now, :program:`wind-utils` can be compiled with :program:`Nalu`
by enabling the compilation flag::

  -DENABLE_WIND_UTILS=ON

during CMake configure phase.
Subsequent make install will install all :program:`wind-utils`
executables along side
:program:`naluX` under the same installation prefix.


The general instructions to run each case
=========================================

  1. Modify the simulation parameters in the :file:`setup.yaml` file.
  2. Execute the :program:`nalu_input_fileX` script with the
     :file:`setup.yaml` file as
     an input.
  3. Generate the mesh using :program:`abl_mesh` from nalu wind utils.
  4. Generate the initial condition using :program:`nalu_preprocess`.
  5. Run the simulation using :program:`naluX`.

.. _examples_environment:

Setting up the environment
==========================

    In order to use the Python utilities to create the input files and
    post-process some of the data, a proper environment needs to be set.
    The user can add these libraries to their Python environment, or use conda
    to create the environment needed.
    Instruction to install conda can be found `here <https://conda.io/docs/user-guide/install/index.html>`_.


    The new environment can be created through conda using::

      conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml netCDF4 matplotlib scipy pandas

    This new environment will allow the execution of
    :program:`nalu_input_fileX`.
    The environment is saved in THE USER system,
    so it needs to be created only once.
    After that, it just needs to be activated.

    Now, to use the environment run::

      source activate nalu_python


The :program:`nalu_input_fileX` script
======================================

  This code is an executable which takes as an input a set-up file.
  The executable will read in the set-up file, and create a new nalu input file
  based on the parameters specified.
  Excuting the code with the -h flag will provide the necessary information::

    ./nalu_input_fileX -h

The :file:`setup.yaml` file
===========================

  This file includes the inputs to be modified for a case.
  This example is for a Neutral Atmospheric Boundary Layer simulation.

  .. literalinclude:: ../../../../examples/abl_neutral/setup.yaml
      :language: yaml
