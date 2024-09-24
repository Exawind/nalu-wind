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
  -DCMAKE_BUILD_TYPE:STRING=RELEASE
  -DMPI_USE_COMPILER_WRAPPERS:BOOL=ON
  -DMPI_CXX_COMPILER:FILEPATH=${CXX}
  -DKokkos_ENABLE_DEPRECATED_CODE:BOOL=OFF
  -DTpetra_INST_SERIAL:BOOL=ON
  -DTrilinos_ENABLE_CXX11:BOOL=ON
  -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON
  -DTpetra_INST_DOUBLE:BOOL=ON
  -DTpetra_INST_COMPLEX_DOUBLE:BOOL=OFF
  -DTrilinos_ENABLE_TESTS:BOOL=OFF
  -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF
  -DTrilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF
  -DTrilinos_ALLOW_NO_PACKAGES:BOOL=OFF
  -DTrilinos_ENABLE_Epetra:BOOL=OFF
  -DTrilinos_ENABLE_Tpetra:BOOL=ON
  -DTrilinos_ENABLE_KokkosKernels:BOOL=ON
  -DTrilinos_ENABLE_ML:BOOL=OFF
  -DTrilinos_ENABLE_MueLu:BOOL=ON
  -DXpetra_ENABLE_Kokkos_Refactor:BOOL=ON
  -DMueLu_ENABLE_Kokkos_Refactor:BOOL=ON
  -DTrilinos_ENABLE_EpetraExt:BOOL=OFF
  -DTrilinos_ENABLE_AztecOO:BOOL=OFF
  -DTrilinos_ENABLE_Belos:BOOL=ON
  -DTrilinos_ENABLE_Ifpack2:BOOL=ON
  -DTrilinos_ENABLE_Amesos2:BOOL=ON
  -DTrilinos_ENABLE_Zoltan2:BOOL=ON
  -DTrilinos_ENABLE_Ifpack:BOOL=OFF
  -DTrilinos_ENABLE_Amesos:BOOL=OFF
  -DTrilinos_ENABLE_Zoltan:BOOL=ON
  -DTrilinos_ENABLE_STK:BOOL=ON
  -DTrilinos_ENABLE_Gtest:BOOL=ON
  -DTrilinos_ENABLE_SEACASExodus:BOOL=ON
  -DTrilinos_ENABLE_SEACASEpu:BOOL=ON
  -DTrilinos_ENABLE_SEACASExodiff:BOOL=ON
  -DTrilinos_ENABLE_SEACASNemspread:BOOL=ON
  -DTrilinos_ENABLE_SEACASNemslice:BOOL=ON
  -DTrilinos_ENABLE_SEACASIoss:BOOL=ON
  -DTPL_ENABLE_MPI:BOOL=ON
  -DTPL_ENABLE_Boost:BOOL=ON
  -DBoostLib_INCLUDE_DIRS:PATH=${BOOST_ROOT_DIR}/include
  -DBoostLib_LIBRARY_DIRS:PATH=${BOOST_ROOT_DIR}/lib
  -DBoost_INCLUDE_DIRS:PATH=${BOOST_ROOT_DIR}/include
  -DBoost_LIBRARY_DIRS:PATH=${BOOST_ROOT_DIR}/lib
  -DTPL_ENABLE_SuperLU:BOOL=ON
  -DSuperLU_INCLUDE_DIRS:PATH=${SUPERLU_ROOT_DIR}/include
  -DSuperLU_LIBRARY_DIRS:PATH=${SUPERLU_ROOT_DIR}/lib
  -DTPL_ENABLE_Netcdf:BOOL=ON
  -DNetCDF_ROOT:PATH=${NETCDF_ROOT_DIR}
  -DTPL_Netcdf_PARALLEL:BOOL=ON
  -DTPL_ENABLE_Pnetcdf:BOOL=ON
  -DPNetCDF_ROOT:PATH=${PARALLEL_NETCDF_ROOT_DIR}
  -DPnetcdf_INCLUDE_DIRS:PATH=${PARALLEL_NETCDF_ROOT_DIR}/include
  -DPnetcdf_LIBRARY_DIRS:PATH=${PARALLEL_NETCDF_ROOT_DIR}/lib
  -DTPL_ENABLE_HDF5:BOOL=ON
  -DHDF5_ROOT:PATH=${HDF5_ROOT_DIR}
  -DHDF5_NO_SYSTEM_PATHS:BOOL=ON
  -DTPL_ENABLE_Zlib:BOOL=ON
  -DZlib_INCLUDE_DIRS:PATH=${ZLIB_ROOT_DIR}/include
  -DZlib_LIBRARY_DIRS:PATH=${ZLIB_ROOT_DIR}/lib
  -DTPL_ENABLE_BLAS:BOOL=ON
  $EXTRA_ARGS \
  ../
