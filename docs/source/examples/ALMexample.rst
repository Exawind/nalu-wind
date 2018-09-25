Actuator Line Model in Uniform Inflow
-------------------------------------

This case is a large-eddy simulation of a wind turbine under uniform inflow.
The wind turbine aerodynamic forces are computed using OpenFAST.


Step by step instructions to run the case
=========================================

  1. Go to the directory where the case is::

      cd nalu-wind/examples/abl_neutral/

  2. Modify the ``set_up.yaml`` file to include all simulation parameters
     necessary.

  3. Generate the new input files.
     First, load the python environment::

      source activate nalu_python

    If the python environment does not exist, then create it first, and then
    activate it::

      conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml
      source activate nalu_python

    Run the exectuable and provide the ``set_up.yaml`` file as input::

      ../nalu_input_fileX -s set_up.yaml

  4. Generate the mesh::

      abl_mesh -i abl_preprocess.yaml

  5. Generate the initial condition::

      abl_preprocess -i abl_preprocess.yaml

  6. Run the nalu executable::

      mpirun -np 600 naluX -i abl_simulation.yaml


Post-processing
===============
