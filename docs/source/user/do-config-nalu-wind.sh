#!/bin/bash -l

# Instructions:
# Make a directory in the Nalu-Wind directory for building,
# Copy this script to that directory and edit the
# options below to your own needs and run it.

CXX_COMPILER=mpicxx
C_COMPILER=mpicc
FORTRAN_COMPILER=mpifort
FLAGS="-O2 -march=native -mtune=native"
OVERSUBSCRIBE_FLAGS="--use-hwthread-cpus --oversubscribe"
  
set -e

TRILINOS_ROOT_DIR=${NALU_ROOT_DIR}/install/trilinos
YAML_CPP_ROOT_DIR=${NALU_ROOT_DIR}/install/yaml-cpp

# Clean before cmake configure
set +e
rm -rf CMakeFiles
rm -f CMakeCache.txt
set -e

# Extra TPLs that can be included in the cmake configure:
#  -DENABLE_PARAVIEW_CATALYST:BOOL=ON \
#  -DPARAVIEW_CATALYST_INSTALL_PATH:PATH=${CATALYST_IOSS_ADAPTER_ROOT_DIR} \
#  -DENABLE_OPENFAST:BOOL=ON \
#  -DOpenFAST_DIR:PATH=${OPENFAST_ROOT_DIR} \
#  -DENABLE_HYPRE:BOOL=ON \
#  -DHYPRE_DIR:PATH=${HYPRE_ROOT_DIR} \
#  -DENABLE_TIOGA:BOOL=ON \
#  -DTIOGA_DIR:PATH=${TIOGA_ROOT_DIR} \

(set -x; cmake \
  -DCMAKE_CXX_COMPILER:STRING=${CXX_COMPILER} \
  -DCMAKE_CXX_FLAGS:STRING="${FLAGS}" \
  -DCMAKE_C_COMPILER:STRING=${C_COMPILER} \
  -DCMAKE_C_FLAGS:STRING="${FLAGS}" \
  -DCMAKE_Fortran_COMPILER:STRING=${FORTRAN_COMPILER} \
  -DCMAKE_Fortran_FLAGS:STRING="${FLAGS}" \
  -DMPI_CXX_COMPILER:STRING=${CXX_COMPILER} \
  -DMPI_C_COMPILER:STRING=${C_COMPILER} \
  -DMPI_Fortran_COMPILER:STRING=${FORTRAN_COMPILER} \
  -DMPIEXEC_PREFLAGS:STRING="${OVERSUBSCRIBE_FLAGS}" \
  -DTrilinos_DIR:PATH=${TRILINOS_ROOT_DIR} \
  -DYAML_DIR:PATH=${YAML_CPP_ROOT_DIR} \
  -DCMAKE_BUILD_TYPE:STRING=RELEASE \
  -DENABLE_DOCUMENTATION:BOOL=OFF \
  -DENABLE_TESTS:BOOL=ON \
  ..)

(set -x; nice make -j 16)
