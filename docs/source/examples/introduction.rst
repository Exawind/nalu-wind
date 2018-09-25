Introduction
------------

A collection of examples for running large-eddy simulations of wind plant
aerodynamics is available under ``nalu-wind/examples/``.
A ``set_up.yaml`` file is included with all cases.
This file has the parameters that can be modified for each case of interest.

Each case has a default collection of input files in ``./input_files``.
The executable ``nalu_input_fileX`` will take the input files and
modify them according to the inputs set in ``set_up.yaml`` and
generate new and ready to use input files.

The general instructions to run each case
=========================================

  1. Modify the simulation parameters in the ``set_up.yaml`` file.
  2. Exectute the ``nalu_input_fileX`` script with the ``set_up.yaml`` file as
     an input.
  3. Generate the mesh using ``abl_mesh`` from nalu wind utils.
  4. Generate the initial condition using ``nalu_preprocess``.
  5. Run the simulation using ``naluX``.

.. _examples_environment:

Setting up the environment
==========================

    In order to use the ``python`` utilities to create the input files and
    post-process some of the data, a proper environment needs to be set.
    The user can add these libraries to their python environment, or use conda
    to create the environment needed.

    The new environment can be created through conda using::

      conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml netCDF4 matplotlib scipy

    Now, to use the environment run::

      source activate nalu_python

    This new environment will allow the execution of the ``nalu_input_fileX``
    script.
    The environment is saved in THE USER system, so it needs to be created only
    once.
    After that, it just needs to be activated.


The ``nalu_input_fileX`` script
===============================

  This code is an executable which takes as an input a set-up file.
  The executable will read in the set-up file, and create a new nalu input file
  based on the parameters specified.
  Excuting the code with the -h flag will provide the necessary information::

    ./nalu_input_fileX -h

The ``set_up.yaml`` file
========================

  This file includes the inputs to be modified for a case.
  This example is for a Neutral Atmospheric Boundary Layer simulation.

  .. literalinclude:: ../../../examples/abl_neutral/set_up.yaml
      :language: yaml
