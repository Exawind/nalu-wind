Building Nalu-Wind Manually
===========================

Although we recommend installing Nalu-Wind with Spack, if you prefer not to build using Spack, below are instructions which describe the process of building Nalu-Wind by hand. These instructions are an approximation, due to the many differences that can exist across machines.

Linux and OSX
-------------

The instructions for Linux and OSX are mostly the same, except on each OS you may be able to use a package manager to install some dependencies for you. Using Homebrew on OSX is one option listed below. Compilers and MPI are expected to be already installed. If they are not, please follow the OpenMPI build instructions. Currently we are recommending OpenMPI v1.10.7 or MPICH 3.3 and GCC v7.3.0. Start by creating a ``${NALU_ROOT_DIR}`` to work in.

Homebrew
~~~~~~~~

If using OSX, you can install many dependencies using Homebrew. Install `Homebrew <https://github.com/Homebrew/brew>`__ on your local machine and reference the list below for some packages Homebrew can install for you. This allows you to skip the steps describing the build process for each of these applications. You will need to find the location of the applications in which Homebrew has installed them, to use when building Trilinos and Nalu-Wind.

::

    brew install openmpi
    brew install cmake
    brew install libxml2
    brew install boost
    brew tap homebrew/science
    brew install superlu43


CMake v3.12.4
~~~~~~~~~~~~~

CMake is provided `here <http://www.cmake.org/download/>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar xf cmake-3.12.4.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/cmake-3.12.4
    ./configure --prefix=${NALU_ROOT_DIR}/install/cmake
    make
    make install

SuperLU v4.3
~~~~~~~~~~~~

SuperLU is provided `here <http://crd-legacy.lbl.gov/~xiaoye/SuperLU/>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar xf superlu_4.3.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/SuperLU_4.3
    cp MAKE_INC/make.linux make.inc

To find out what the correct platform extension PLAT is:

::

    uname -m

Edit ``make.inc`` as shown below (diffs shown from baseline).

::

    PLAT          = _x86_64
    SuperLUroot   = /your_path/install/SuperLU_4.3 i.e., ${NALU_ROOT_DIR}/install/SuperLU_4.3
    BLASLIB       = -L/usr/lib64 -lblas
    CC            = mpicc
    FORTRAN       = mpif77

On some platforms, the ``${NALU_ROOT_DIR}`` may be mangled. In such cases, you may need to use the entire path to ``install/SuperLU_4.3``.

Next, make some new directories:

::

    mkdir ${NALU_ROOT_DIR}/install/SuperLU_4.3
    mkdir ${NALU_ROOT_DIR}/install/SuperLU_4.3/lib
    mkdir ${NALU_ROOT_DIR}/install/SuperLU_4.3/include
    cd ${NALU_ROOT_DIR}/packages/SuperLU_4.3
    make
    cp SRC/*.h ${NALU_ROOT_DIR}/install/SuperLU_4.3/include

Libxml2 v2.9.8
~~~~~~~~~~~~~~

Libxml2 is found `here <http://www.xmlsoft.org/sources/>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -xvf libxml2-2.9.8.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/libxml2-2.9.8
    CC=mpicc CXX=mpicxx ./configure -without-python --prefix=${NALU_ROOT_DIR}/install/libxml2
    make
    make install

Boost v1.68.0
~~~~~~~~~~~~~

Boost is found `here <http://www.boost.org>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -zxvf boost_1_68_0.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/boost_1_68_0
    ./bootstrap.sh --prefix=${NALU_ROOT_DIR}/install/boost --with-libraries=signals,regex,filesystem,system,mpi,serialization,thread,program_options,exception

Next, edit ``project-config.jam`` and add a 'using mpi', e.g,

using mpi: /path/to/mpi/openmpi/bin/mpicc

::

    ./b2 -j 4 2>&1 | tee boost_build_one
    ./b2 -j 4 install 2>&1 | tee boost_build_intall

YAML-CPP 0.6.2
~~~~~~~~~~~~~~

YAML is provided `here <https://github.com/jbeder/yaml-cpp>`__. Versions of Nalu before v1.1.0 used earlier versions of YAML-CPP. For brevity only the latest build instructions are discussed and the history of the Nalu-Wind git repo can be used to find older installation instructions if required.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    git clone https://github.com/jbeder/yaml-cpp
    cd yaml-cpp && git checkout yaml-cpp-0.6.2

Build:

::

    cd ${NALU_ROOT_DIR}/packages/yaml-cpp
    mkdir build
    cd build
    cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS=-std=c++11 -DCMAKE_CC_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=${NALU_ROOT_DIR}/install/yaml-cpp ..
    make
    make install


Zlib v1.2.11
~~~~~~~~~~~~

Zlib is provided `here <http://www.zlib.net>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -zxvf zlib-1.2.11.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/zlib-1.2.11
    CC=gcc CXX=g++ CFLAGS=-O3 CXXFLAGS=-O3 ./configure --prefix=${NALU_ROOT_DIR}/install/zlib
    make
    make install

HDF5 v1.10.4
~~~~~~~~~~~~

HDF5 1.10.4 is provided `here <http://www.hdfgroup.org/downloads/index.html>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -zxvf hdf5-1.10.4.tar.gz

Build:

::

    cd ${NALU_ROOT_DIR}/packages/hdf5-1.10.4
    ./configure CC=mpicc FC=mpif90 CXX=mpicxx CXXFLAGS="-fPIC -O3" CFLAGS="-fPIC -O3" FCFLAGS="-fPIC -O3" --enable-parallel --with-zlib=${NALU_ROOT_DIR}/install/zlib --prefix=${NALU_ROOT_DIR}/install/hdf5
    make
    make install
    make check
        

NetCDF v4.6.1 and Parallel NetCDF v1.8.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to support all aspects of Nalu-Wind's parallel models, this combination of products is required.

Parallel NetCDF v1.8.0
**********************

Parallel NetCDF is provided on the `Argon Trac Page <https://trac.mcs.anl.gov/projects/parallel-netcdf/wiki/Download>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -zxvf parallel-netcdf-1.8.0.tar.gz

Build:

::

    cd parallel-netcdf-1.8.0
    ./configure --prefix=${NALU_ROOT_DIR}/install/parallel-netcdf CC=mpicc FC=mpif90 CXX=mpicxx CFLAGS=-O3 CXXFLAGS=-O3 --disable-fortran
    make
    make install

NetCDF v4.6.1
*************

NetCDF is provided `here <https://github.com/Unidata/netcdf-c/releases>`__.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    tar -zxvf netcdf-4.6.1.tar.gz

Build:

::

    cd netcdf-4.6.1
    ./configure --prefix=${NALU_ROOT_DIR}/install/netcdf CC=mpicc FC=mpif90 CXX=mpicxx CFLAGS="-I${NALU_ROOT_DIR}/install/parallel-netcdf/include -O3" LDFLAGS=-L${NALU_ROOT_DIR}/install/parallel-netcdf/lib --enable-pnetcdf --enable-parallel-tests --enable-netcdf-4 --disable-shared --disable-fsync --disable-cdmremote --disable-dap --disable-doxygen --disable-v2
    make -j 4 
    make check
    make install


Trilinos
~~~~~~~~

Trilinos is managed by the `Trilinos <http://www.trilinos.org>`__ project and can be found on Github.

Prepare:

::

    cd ${NALU_ROOT_DIR}/packages
    git clone https://github.com/trilinos/Trilinos.git
    cd ${NALU_ROOT_DIR}/packages/Trilinos
    mkdir build
    cd build

Now create a ``do-config-trilinos`` script with the following recommended options:

.. literalinclude:: do-config-trilinos.sh

Build
*****

Place into the build directory, the ``do-config-trilinos`` script created from the recommended Trilinos configuration listed above.

``do-config-trilinos`` will be used to run cmake to build trilinos correctly for Nalu-Wind.

Make sure all other paths to netcdf, hdf5, etc., are correct.

::

    ./do-config-trilinos
    make
    make install

HYPRE
~~~~~

Nalu-Wind can use HYPRE solvers and preconditioners, especially for Pressure Poisson
solves. However, this dependency is optional and is not enabled by default.
Users wishing to use HYPRE solver and preconditioner combination must compile
HYPRE library and link to it when building Nalu-Wind.

.. code-block:: bash

   # 1. Clone hypre sources
   https://github.com/LLNL/hypre.git
   cd hypre/src

   # 2. Configure HYPRE package and pass installation directory
   ./configure --prefix=${NALU_ROOT_DIR}/install/hypre --without-superlu --without-openmp --enable-bigint

   # 3. Compile and install
   make && make install

.. note::

   #. Make sure that ``--enable-bigint`` option is turned on if you intend to
      run linear systems with :math:`> 2` billion rows. Otherwise, ``nalu``
      executable will throw an error at runtime for large problems.

   #. Users must pass ``-DENABLE_HYPRE`` option to CMake during Nalu-Wind
      configuration phase. Optionally, the variable `-DHYPRE_DIR`` can be used
      to pass the path of HYPRE install location to CMake.


ParaView Catalyst
~~~~~~~~~~~~~~~~~

Optionally enable `ParaView Catalyst <https://www.paraview.org/in-situ/>`__
for in-situ visualization with Nalu-Wind. These instructions can be skipped if 
you do not require in-situ visualization with Nalu-Wind. The first thing you 
will need to do is build Paraview yourself using their SuperBuild instructions.

Build Nalu-Wind ParaView Catalyst Adapter
*****************************************

Next you will need to build the Catalyst adapter for Trilinos to hook into Paraview.
The adapter is located in the Trilinos repo at ``Trilinos/packages/seacas/libraries/ioss/src/visualization/ParaViewCatalystIossAdapter``. To install:

::

    cd Trilinos/packages/seacas/libraries/ioss/src/visualization/ParaViewCatalystIossAdapter
    cmake -DParaView_DIR:PATH=/path/to/paraview/lib/cmake/paraview_version -DCMAKE_INSTALL_PREFIX:PATH=${NALU_ROOT_DIR}/install
    make
    make install

Nalu-Wind
~~~~~~~~~

Nalu-Wind is provided `here <https://github.com/exawind/nalu-wind>`__. The master branch of Nalu-Wind typically matches with the master branch or develop branch of Trilinos. If it is necessary to build an older version of Nalu-Wind, refer to the history of the Nalu git repo for instructions on doing so.

Prepare:

::

    git clone https://github.com/Exawind/nalu-wind.git


Build
*****

Create a ``nalu-wind/build`` directory and execute something similar to following commands. The general commands for configuring and building Nalu-Wind are listed below. We show a script which uses modules which populate the <PACKAGE>_ROOT_DIR locations for the NREL Eagle machine, but it will need to be modified with the specific TPL locations you have used.

.. literalinclude:: do-config-nalu-wind.sh

This process will create ``naluX`` within the ``nalu-wind/build`` location.

