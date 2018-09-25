Atmospheric Boundary Layer - Neutral
------------------------------------

This case is a large-eddy simulation of a neutral atmospheric boundary layer.
The case uses periodic boundary conditions on the sides (east, west, north,
south), a wall-model in the bottom wall and a stress free boundary condition at
the top of the domain.
A PI controller is used to drive the velocity at a given height.
The PI controller adjusts the forcing at each time-step to match a given planar
average velocity at a given height.
It takes about 10,000 [s] for the the turbulence to develop.
The example runs for 20,000 [s].

Step by step instructions to run the case
=========================================

  1. Go to the directory where the case is::

      cd nalu-wind/examples/abl_neutral/

  2. Modify the ``set_up.yaml`` file to include all the necessary simulation
     parameters.

  3. Generate the new input files.
     First, load the python environment::

      source activate nalu_python

    If the python environment does not exist, create it first, and then activate
    it.
    Instructions for creating the environment are provided in
    :ref:`examples_environment`.
    Run the exectuable and provide the ``set_up.yaml`` file as input::

      ../nalu_input_fileX -s set_up.yaml

  4. Generate the mesh::

      abl_mesh -i abl_preprocess.yaml

  5. Generate the initial condition::

      abl_preprocess -i abl_preprocess.yaml

  6. Run the nalu executable::

      mpirun -np 600 naluX -i abl_simulation.yaml

     In this example 600 processors are used, but any number of processors could
     be used.
     Target 50K elements per core for choosing number of MPI cores.


Post-processing
===============

Nalu's output is generated at runtime. The ``plot_abl.py`` pythin script is used
to plot simulation results. The script will load the plane-averaged statistics
and plot them as function of time and height.
To run the script, load the python environment, and then run::

  source activate nalu_python
  python plot_abl.py
