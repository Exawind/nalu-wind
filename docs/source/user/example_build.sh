#!/bin/bash

# The base directory where mpi is located.
# From here you should be able to find include/mpi.h bin/mpicxx, bin/mpiexec, etc.
mpi_base_dir=/PathToMPI
nalu_build_dir=/PathToScratchBuild

# Note: Don't forget to set your LD_LIBRARY_PATH to $mpi_base_dir/lib
#       You may also need to add to LD_LIBRARY_PATH the lib directory for the compiler
#       used to create the mpi executables.

# TPLS needed by trilinos, possibly provided by HomeBrew on a Mac
#boost_dir=/usr/local/Cellar/boost/1.56.0/include/boost/
#superlu_inc_dir=/usr/local/Cellar/superlu/4.3/include/superlu
#superlu_lib_dir=/usr/local/Cellar/superlu/4.3/lib
boost_dir=$nalu_build_dir/install
superlu_inc_dir=$nalu_build_dir/install/SuperLU_4.3/include
superlu_lib_dir=$nalu_build_dir/install/SuperLU_4.3/lib

# Additional needed TPLS
netcdf_install_dir=$nalu_build_dir/install
hdf_install_dir=$nalu_build_dir/install
pnetcdf_install_dir=$nalu_install_dir
z_install_dir=$nalu_build_dir/install

# Where trilinos will be installed
trilinos_install_dir=$nalu_build_dir/install/trilinos

echo "nalu_build_dir = \"$nalu_build_dir\""

EXTRA_ARGS=$@

# Cleanup old cache before we configure
# Note:  This does not remove files produced by make.  Use "make clean" for this.
find . -name "CMakeFiles" -exec rm -rf {} \;
rm -f CMakeCache.txt

cmake \
-DCMAKE_INSTALL_PREFIX=$trilinos_install_dir \
-DTrilinos_ENABLE_CXX11=ON \
-DCMAKE_BUILD_TYPE=RELEASE \
-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-DTpetra_INST_DOUBLE:BOOL=ON \
-DTpetra_INST_INT_LONG:BOOL=ON \
-DTpetra_INST_COMPLEX_DOUBLE=OFF \
-DTrilinos_ENABLE_TESTS:BOOL=OFF \
-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-DTrilinos_ALLOW_NO_PACKAGES:BOOL=OFF \
-DTPL_ENABLE_MPI=ON \
-DMPI_BASE_DIR:PATH=$mpi_base_dir \
-DTPL_ENABLE_SuperLU=ON \
-DSuperLU_INCLUDE_DIRS:PATH=$superlu_inc_dir \
-DSuperLU_LIBRARY_DIRS:PATH=$superlu_lib_dir \
-DTrilinos_ENABLE_Epetra:BOOL=OFF \
-DTrilinos_ENABLE_Tpetra:BOOL=ON \
-DTrilinos_ENABLE_ML:BOOL=OFF \
-DTrilinos_ENABLE_MueLu:BOOL=ON \
-DTrilinos_ENABLE_EpetraExt:BOOL=OFF \
-DTrilinos_ENABLE_AztecOO:BOOL=OFF \
-DTrilinos_ENABLE_Belos:BOOL=ON \
-DTrilinos_ENABLE_Ifpack2:BOOL=ON \
-DTrilinos_ENABLE_Amesos2:BOOL=ON \
-DTrilinos_ENABLE_Zoltan2:BOOL=ON \
-DTrilinos_ENABLE_Ifpack:BOOL=OFF \
-DTrilinos_ENABLE_Amesos:BOOL=OFF \
-DTrilinos_ENABLE_Zoltan:BOOL=ON \
-DTrilinos_ENABLE_STKMesh:BOOL=ON \
-DTrilinos_ENABLE_STKSimd:BOOL=ON \
-DTrilinos_ENABLE_STKIO:BOOL=ON \
-DTrilinos_ENABLE_STKTransfer:BOOL=ON \
-DTrilinos_ENABLE_STKSearch:BOOL=ON \
-DTrilinos_ENABLE_STKUtil:BOOL=ON \
-DTrilinos_ENABLE_STKTopology:BOOL=ON \
-DTrilinos_ENABLE_STKUnit_tests:BOOL=ON \
-DTrilinos_ENABLE_STKUnit_test_utils:BOOL=ON \
-DTrilinos_ENABLE_Gtest:BOOL=ON \
-DTrilinos_ENABLE_STKClassic:BOOL=OFF \
-DTrilinos_ENABLE_SEACASExodus:BOOL=ON \
-DTrilinos_ENABLE_SEACASEpu:BOOL=ON \
-DTrilinos_ENABLE_SEACASExodiff:BOOL=ON \
-DTrilinos_ENABLE_SEACASNemspread:BOOL=ON \
-DTrilinos_ENABLE_SEACASNemslice:BOOL=ON \
-DTrilinos_ENABLE_SEACASIoss:BOOL=ON \
-DTPL_ENABLE_Netcdf:BOOL=ON \
-DNetCDF_ROOT:PATH=${netcdf_install_dir} \
-DTPL_ENABLE_HDF5:BOOL=ON \
-DHDF5_ROOT:PATH=${hdf_install_dir} \
-DHDF5_NO_SYSTEM_PATHS=ON \
-DPNetCDF_ROOT:PATH=${pnetcdf_install_dir} \
-DZlib_ROOT:PATH=${z_install_dir} \
-DBoostLib_INCLUDE_DIRS:PATH="$boost_dir/include" \
-DBoostLib_LIBRARY_DIRS:PATH="$boost_dir/lib" \
-DTrilinos_ASSERT_MISSING_PACKAGES=OFF \
$EXTRA_ARGS \
../
