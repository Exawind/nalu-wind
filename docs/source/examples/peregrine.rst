Peregrine
---------

Here are the instructions to use Nalu and these examples on NREL's HPC system
Peregrine.

  1. Create the conda environment::

      module load conda
      conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml netCDF4 matplotlib

  2. Set the Nalu environment.
     This can be done by adding the following function to your ``.bash_profile``::

       function NaluEnv {
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
           esac}


  3. Load the nalu environment::

      NaluEnv

The system is now ready to compile and use Nalu.
