#!/bin/bash

# The base directory where mpi is located.
# From here you should be able to find include/mpi.h bin/mpicxx, bin/mpiexec, etc.
MPI_ROOT_DIR=/PathToMPI
NALU_ROOT_DIR=/PathToNaluProjectDir

# Note: Don't forget to set your LD_LIBRARY_PATH to $mpi_base_dir/lib
#       You may also need to add to LD_LIBRARY_PATH the lib directory for the compiler
#       used to create the mpi executables.

# TPLS needed by trilinos, possibly provided by HomeBrew on a Mac
#BOOST_ROOT_DIR=/usr/local/Cellar/boost/1.56.0/include/boost/
#SUPERLU_ROOT_DIR=/usr/local/Cellar/superlu/4.3
BOOST_ROOT_DIR=${NALU_BUILD_DIR}/install/boost
SUPERLU_ROOT_DIR=${NALU_BUILD_DIR}/install/SuperLU_4.3
NETCDF_ROOT_DIR=${NALU_BUILD_DIR}/install/netcdf
HDF5_ROOT_DIR=${NALU_BUILD_DIR}/install/hdf5
PARALLEL_NETCDF_ROOT_DIR=${NALU_BUILD_DIR}/install/parallel-netcdf
ZLIB_ROOT_DIR=${NALU_BUILD_DIR}/install/zlib
TRILINOS_ROOT_DIR=${NALU_BUILD_DIR}/install/trilinos

EXTRA_ARGS=$@

# Cleanup old cache before we configure
# Note:  This does not remove files produced by make.  Use "make clean" for this.
find . -name "CMakeFiles" -exec rm -rf {} \;
rm -f CMakeCache.txt

cmake \
-DCMAKE_INSTALL_PREFIX=${TRILINOS_ROOT_DIR} \
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
-DMPI_BASE_DIR:PATH=${MPI_ROOT_DIR} \
-DTPL_ENABLE_SuperLU=ON \
-DSuperLU_INCLUDE_DIRS:PATH=${SUPERLU_ROOT_DIR/include/superlu \
-DSuperLU_LIBRARY_DIRS:PATH=${SUPERLU_ROOT_DIR}/lib \
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
-DNetCDF_ROOT:PATH=${NETCDF_ROOT_DIR} \
-DTPL_ENABLE_HDF5:BOOL=ON \
-DHDF5_ROOT:PATH=${HDF5_ROOT_DIR} \
-DHDF5_NO_SYSTEM_PATHS=ON \
-DPNetCDF_ROOT:PATH=${PARALLEL_NETCDF_ROOT_DIR} \
-DZlib_ROOT:PATH=${ZLIB_ROOT_DIR} \
-DBoostLib_INCLUDE_DIRS:PATH="${BOOST_ROOT_DIR}/include" \
-DBoostLib_LIBRARY_DIRS:PATH="${BOOST_ROOT_DIR}/lib" \
-DTrilinos_ASSERT_MISSING_PACKAGES=OFF \
$EXTRA_ARGS \
../
