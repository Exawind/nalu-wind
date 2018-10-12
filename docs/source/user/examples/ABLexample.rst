.. _abl_neutral_example:

Atmospheric Boundary Layer - Neutral
------------------------------------

This case is a large-eddy simulation of a neutral atmospheric boundary layer.
The case uses periodic boundary conditions on the sides (east, west, north,
south), a wall-model in the bottom wall and a stress free boundary condition at
the top of the domain.
A proportional controller is used to drive the velocity at a given height.
The controller adjusts the forcing at each time-step to match a
given planar average velocity at a given height.
More information about the controller can be found
in :ref:`abl_forcing_term`.
It takes about 10,000 [s] for the the turbulence to develop.
The example runs for 20,000 [s].

Step by step instructions to run the case
=========================================

  1. Load the appropriate Nalu environment.
     This requires loading the libraries and Python environment as described in
     :ref:`examples_environment`.
     For users on Peregrine the function defined in :ref:`peregrine_environment`
     should suffice::

       nalu_env

  2. Go to the directory where the case is::

      cd nalu-wind/examples/abl_neutral/

  3. Modify the ``setup.yaml`` file to include all the necessary simulation
     parameters.

  4. Run the executable and provide the ``setup.yaml`` file as input::

      ../nalu_input_fileX -s setup.yaml

     For users on Peregrine, now copy the executables to the case directory::

      cp /projects/windsim/nalu-wind-executables/* .

  5. Generate the mesh::

      ./abl_mesh -i abl_preprocess.yaml

  6. Generate the initial condition::

      ./nalu_preprocess -i abl_preprocess.yaml

  7. Run the nalu executable::

      mpirun -np 600 naluX -i abl_simulation.yaml

     In this example 600 processors are used, but any number of processors could
     be used.
     Target 50K elements per core for choosing number of MPI cores.


Post-processing
===============

Nalu's output is generated at runtime. The ``abl_plots.py`` Python script
is used to plot simulation results.
The script will load the plane-averaged statistics
and plot them as function of time and height.
To run the script, load the Python environment if needed, and run the Python
script::

  python abl_plots.py
