Peregrine
---------

Here are the instructions to use Nalu and these examples on NREL's HPC system
Peregrine.

.. _peregrine_environment:

Initial Setup
=============

These steps need to be completed only once to setup the appropriate Nalu
environment on the Peregrine system.

  1. Create the conda environment::

      module load conda
      conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml netCDF4 matplotlib scipy pandas

  2. Set the Nalu environment. This is the same environment used to compile
     Nalu. If this has not been set, then it can be done by adding the following
     function to ``${HOME}/.bash_profile``::

      function nalu_env {
           module purge
           # Load the python environment
           module load conda
           source activate nalu_python

           local mod_dir=/nopt/nrel/ecom/ecp/base/modules/
           module use ${mod_dir}/gcc-6.2.0
           module load gcc/6.2.0

           compiler=${1:-gcc}

           case ${compiler} in
              gcc)
                  module unuse ${mod_dir}/intel-18.1.163
                  module load binutils openmpi/3.1.1 netlib-lapack cmake
                  ;;
              intel)
                  module load intel-parallel-studio/cluster.2018.1
                  module use ${mod_dir}/intel-18.1.163
                  module load binutils intel-mpi intel-mkl cmake
                  ;;
           esac
       }

     Source the new ``${HOME}/.bash_profile``::

       source ${HOME}/.bash_profile

  3. Clone the Nalu repository by::

       git clone https://github.com/Exawind/nalu-wind.git

Now the environment is ready to run the examples.
To get started, go to: :ref:`abl_neutral_example`.

Running Every Case
==================

Every time the user logs into Peregrine and wants to run a case, these steps
need to be completed:

  1. Load the nalu environment::

      nalu_env

  2. Copy the executables from the public location to the directory where the
     case is::

      cp /projects/windsim/nalu-wind-executables/* .


The system is now ready to compile and use Nalu.
