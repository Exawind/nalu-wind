Actuator Line Model in Uniform Inflow
-------------------------------------

This case is a large-eddy simulation of a wind turbine under uniform inflow.
The wind turbine aerodynamic forces are computed using OpenFAST.


Step by step instructions to run the case
=========================================

  1. Go to the directory where the case is::

      cd nalu-wind/examples/turbine_uniform_inflow/

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

      abl_mesh -i alm_preprocess.yaml

  6. Run the nalu executable::

      mpirun -np 600 naluX -i alm_simulation.yaml

     In this example 600 processors are used, but any number of processors could
     be used.
     Target 50K elements per core for choosing number of MPI cores.


Post-processing
===============
