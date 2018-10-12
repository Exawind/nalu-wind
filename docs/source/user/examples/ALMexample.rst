.. _alm_example:


Actuator Line Model in Uniform Inflow
-------------------------------------

This case is a large-eddy simulation of 2 aligned wind turbines under
uniform inflow.
The turbines are represented using an actuator line model.
The turbine model is the NREL 5MW Reference.
The case has uniform inflow boundary condition with an outflow boundary
condition on the other end.
The sides are specified as zero stress boundary conditions.
The wind turbine aerodynamic forces are computed using OpenFAST.


Step by step instructions to run the case
=========================================

1. Load the appropriate Nalu environment.
   This requires loading the libraries and Python environment as described in
   :ref:`examples_environment`.
   For users on Peregrine the function defined in :ref:`peregrine_environment`
   should suffice::

     nalu_env

2. Go to the directory where the case is::

    cd nalu-wind/examples/turbine_uniform_inflow/

3. Modify the ``setup.yaml`` file to include all the necessary simulation
   parameters.

4. Run the executable and provide the ``setup.yaml`` file as input::

    ../nalu_input_fileX -s setup.yaml

   For users on Peregrine, now copy the executables to the case directory::

    cp /projects/windsim/nalu-wind-executables/* .

5. Generate the mesh::

    ./abl_mesh -i alm_preprocess.yaml

6. Generate the initial condition::

    ./nalu_preprocess -i alm_preprocess.yaml

7. Run the nalu executable::

    mpirun -np 24 naluX -i alm_simulation.yaml

Post-processing
===============

The turbine output is generated at runtime.
The ``plot_alm.py`` Python script
is used to plot turbine output.
The script will load the OpenFAST data
and plot it as a function of time.
To run the script, load the Python environment if needed, and run the Python
script::

  python plot_alm.py
