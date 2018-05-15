Building Nalu-Wind Manually
======================

If you prefer not to build using Spack, below are instructions which describe the process of building Nalu-Wind by hand.

Linux and OSX
-------------

The instructions for Linux and OSX are mostly the same, except on each OS you may be able to use a package manager to install some dependencies for you. Using Homebrew on OSX is one option listed below. Compilers and MPI are expected to be already installed. If they are not, please follow the open-mpi build instructions. Below, we are using OpenMPI v1.10.4 and GCC v4.9.2. Start by create a ``$nalu_build_dir`` to work in.

Homebrew
~~~~~~~~

If using OSX, you can install many dependencies using Homebrew. Install `Homebrew <https://github.com/Homebrew/homebrew/wiki/Installation>`__ on your local machine and reference the list below for some packages Homebrew can install for you which allows you to skip the steps describing the build process for each application, but not that you will need to find the location of the applications in which Homebrew has installed them, to use when building Trilinos and Nalu-Wind.

::

    brew install openmpi
    brew install cmake
    brew install libxml2
    brew install boost
    brew tap homebrew/science
    brew install superlu43


CMake v3.9.4
~~~~~~~~~~~~

CMake is provided `here <http://www.cmake.org/download/>`__.

Prepare:

::

    cd $nalu_build_dir/packages
    tar xf cmake-3.9.4.tar.gz

Build:

::

    cd $nalu_build_dir/packages/cmake-3.9.4
    ./configure --prefix=$nalu_build_dir/install
    make
    make install

SuperLU v4.3
~~~~~~~~~~~~

SuperLU is provided `here <http://crd-legacy.lbl.gov/~xiaoye/SuperLU/>`__.

Prepare:

::

    cd $nalu_build_dir/packages
    tar xf superlu_4.3.tar.gz

Build:

::

    cd $nalu_build_dir/packages/SuperLU_4.3
    cp MAKE_INC/make.linux make.inc

To find out what the correct platform extension PLAT is:

::

    uname -m

Edit ``make.inc`` as shown below (diffs shown from baseline).

::

    PLAT = _x86_64
    SuperLUroot   = /your_path/install/SuperLU_4.3 i.e., $nalu_build_dir/install/SuperLU_4.3
    BLASLIB       = -L/usr/lib64 -lblas
    CC            = mpicc
    FORTRAN       = mpif77

On some platforms, the ``$nalu_build_dir`` may be mangled. In such cases, you may need to use the entire path to ``install/SuperLU_4.3``.

Next, make some new directories:

::

    mkdir $nalu_build_dir/install/SuperLU_4.3
    mkdir $nalu_build_dir/install/SuperLU_4.3/lib
    mkdir $nalu_build_dir/install/SuperLU_4.3/include
    cd $nalu_build_dir/packages/SuperLU_4.3
    make
    cp SRC/*.h $nalu_build_dir/install/SuperLU_4.3/include

Libxml2 v2.9.4
~~~~~~~~~~~~~~

Libxml2 is found `here <http://www.xmlsoft.org/sources/>`__.

Prepare:

::

    cd $nalu_build_dir/packages
    tar -xvf libxml2-2.9.4.tar.gz

Build:

::

    cd $nalu_build_dir/packages/libxml2-2.9.4
    CC=mpicc CXX=mpicxx ./configure -without-python --prefix=$nalu_build_dir/install
    make
    make install

Boost v1.66.0
~~~~~~~~~~~~~

Boost is found `here <http://www.boost.org>`__.

Prepare:

::

    cd $nalu_build_dir/packages
    tar -zxvf boost_1_66_0.tar.gz

Build:

::

    cd $nalu_build_dir/packages/boost_1_66_0
    ./bootstrap.sh --prefix=$nalu_build_dir/install --with-libraries=signals,regex,filesystem,system,mpi,serialization,thread,program_options,exception

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

    cd $nalu_build_dir/packages
    git clone https://github.com/jbeder/yaml-cpp
    cd yaml-cpp && git checkout yaml-cpp-0.6.2

Build:

::

    cd $nalu_build_dir/packages/yaml-cpp
    mkdir build
    cd build
    cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS=-std=c++11 -DCMAKE_CC_COMPILER=mpicc -DCMAKE_INSTALL_PREFIX=$nalu_build_dir/install ..
    make
    make install


Zlib v1.2.11
~~~~~~~~~~~~

Zlib is provided `here <http://www.zlib.net>`__.

Prepare:

::

    cd $nalu_build_dir/packages
    tar -zxvf zlib-1.2.11.tar.gz

Build:

::

    cd $nalu_build_dir/packages/zlib-1.2.11
    CC=gcc CXX=g++ CFLAGS=-O3 CXXFLAGS=-O3 ./configure --prefix=$nalu_build_dir/install/
    make
    make install

HDF5 v1.10.1
~~~~~~~~~~~~

HDF5 1.10.1 is provided `here <http://www.hdfgroup.org/downloads/index.html>`__.

Prepare:

::

    cd $nalu_build_dir/packages/
    tar -zxvf hdf5-1.10.1.tar.gz

Build:

::

    cd $nalu_build_dir/packages/hdf5-1.10.1
    ./configure CC=mpicc FC=mpif90 CXX=mpicxx CXXFLAGS="-fPIC -O3" CFLAGS="-fPIC -O3" FCFLAGS="-fPIC -O3" --enable-parallel --with-zlib=$nalu_build_dir/install --prefix=$nalu_build_dir/install
    make
    make install
    make check
        

NetCDF v4.4.1.1 and Parallel NetCDF v1.8.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to support all aspects of Nalu-Wind's parallel models, this combination of products is required.

Parallel NetCDF v1.8.0
**********************

Parallel NetCDF is provided on the `Argon Trac Page <https://trac.mcs.anl.gov/projects/parallel-netcdf/wiki/Download>`__.

Prepare:

::

    cd $nalu_build_dir/packages/
    tar -zxvf parallel-netcdf-1.8.0.tar.gz

Build:

::

    cd parallel-netcdf-1.8.0
    ./configure --prefix=$nalu_install_dir CC=mpicc FC=mpif90 CXX=mpicxx CFLAGS="-I$nalu_install_dir/include -O3" LDFLAGS=-L$nalu_install_dir/lib --disable-fortran
    make
    make install

Note that we have created an install directory that might look like ``$nalu_build_dir/install``.

NetCDF v4.4.1.1
***************

NetCDF is provided `here <https://github.com/Unidata/netcdf-c/releases>`__.

Prepare:

::

    cd $nalu_build_dir/packages/
    tar -zxvf netcdf-c-4.4.1.1.tar.gz

Build:

::

    cd netcdf-c-4.4.1.1
    ./configure --prefix=$nalu_install_dir CC=mpicc FC=mpif90 CXX=mpicxx CFLAGS="-I$nalu_install_dir/include -O3" LDFLAGS=-L$nalu_install_dir/lib --enable-pnetcdf --enable-parallel-tests --enable-netcdf-4 --disable-shared --disable-fsync --disable-cdmremote --disable-dap --disable-doxygen --disable-v2
    make -j 4 
    make check
    make install


Trilinos
~~~~~~~~

Trilinos is managed by the `Trilinos <http://www.trilinos.org>`__ project and can be found on Github.

Prepare:

::

    cd $nalu_build_dir/packages/
    git clone https://github.com/trilinos/Trilinos.git
    cd $nalu_build_dir/packages/Trilinos
    mkdir build


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
   ./configure --prefix=$nalu_install_dir --without-superlu --without-openmp --enable-bigint

   # 3. Compile and install
   make && make install

.. note::

   #. Make sure that ``--enable-bigint`` option is turned on if you intend to
      run linear systems with :math:`> 2` billion rows. Otherwise, ``nalu``
      executable will throw an error at runtime for large problems.

   #. Users must pass ``-DENABLE_HYPRE`` option to CMake during Nalu-Wind
      configuration phase. Optionally, the variable `-DHYPRE_DIR`` can be used
      to pass the path of HYPRE install location to CMake.

Build
*****

Place into the build directory, one of the ``do-configTrilinos_*`` files, that can be obtained from the Nalu-Wind repo.

``do-configTrilinos_*`` will be used to run cmake to build trilinos correctly for Nalu-Wind. Note that there are two files: one for 'release' and the other 'debug'. The files can be found on the Nalu-Wind GitHub site or copied from ``$nalu_build_dir/packages/nalu-wind/build``, which is created in the Nalu-Wind build step documented below. For example:

Pull latest version of ``do-configTrilinos_*`` from Nalu-Wind's GitHub site:

::

    curl -o $nalu_build_dir/packages/Trilinos/build/do-configTrilinos_release https://github.com/Exawind/nalu-wind/blob/master/build/do-configTrilinos_release

Or if you create the Nalu-Wind directory as directed below, simply copy one of the ``do-configTrilinos_*`` files from local copy of Nalu-Wind's git repository:

::

    cp $nalu_build_dir/packages/nalu-wind/build/do-configTrilinos_release $nalu_build_dir/packages/Trilinos/build

Now edit ``do-configTrilinos_release`` to modify the paths so they point to ``$nalu_build_dir/install``.

::

    cd $nalu_build_dir/packages/Trilinos/build
    chmod +x do-configTrilinos_release

Make sure all other paths to netcdf, hdf5, etc., are correct.

::

    ./do-configTrilinos_release
    make
    make install


ParaView Catalyst
~~~~~~~~~~~~~~~~~

Optionally enable `ParaView Catalyst <https://www.paraview.org/in-situ/>`__
for in-situ visualization with Nalu-Wind. These instructions can be skipped if 
you do not require in-situ visualization with Nalu-Wind.

Build ParaView SuperBuild v5.3.0
********************************

The `ParaView SuperBuild <https://gitlab.kitware.com/paraview/paraview-superbuild>`__ 
builds ParaView along with all dependencies necessary to enable Catalyst with Nalu-Wind.
Clone the ParaView SuperBuild within ``$nalu_build_dir/packages``:

::

    cd $nalu_build_dir/packages/
    git clone --recursive https://gitlab.kitware.com/paraview/paraview-superbuild.git
    cd paraview-superbuild
    git fetch origin
    git checkout v5.3.0
    git submodule update

Create a new build folder in ``$nalu_build_dir/``:

::

    cd $nalu_build_dir
    mkdir paraview-superbuild-build
    cd paraview-superbuild-build

Copy ``do-configParaViewSuperBuild`` to ``paraview-superbuild-build``.
Edit ``do-configParaViewSuperBuild`` to modify the defined paths as
follows:

::

    mpi_base_dir=<same MPI base directory used to build Trilinos>
    nalu_build_dir=<path to root nalu build dir>

Make sure the MPI library names are correct.

::

    ./do-configParaViewSuperBuild
    make -j 8
   
Build Nalu-Wind ParaView Catalyst Adapter
*****************************************

Create a new build folder in ``$nalu_build_dir/``:

::

    cd $nalu_build_dir
    mkdir nalu-catalyst-adapter-build
    cd nalu-catalyst-adapter-build

Copy ``do-configNaluCatalystAdapter`` to ``nalu-catalyst-adapter-build``.
Edit ``do-configNaluCatalystAdapter`` and modify ``nalu_build_dir`` at the
top of the file to the root build directory path.

::

    ./do-configNaluCatalystAdapter
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

In ``Nalu-Wind/build``, you will find the `do-configNalu <https://github.com/Exawind/nalu-wind/blob/master/build/do-configNalu_release>`__ script. Copy the ``do-configNalu_release`` or ``debug`` file to a new, non-tracked file:

::

    cp do-configNalu_release do-configNaluNonTracked

Edit the paths at the top of the files by defining the ``nalu_build_dir variable``. Within ``Nalu-Wind/build``, execute the following commands:

::

    ./do-configNaluNonTracked
    make 

This process will create ``naluX`` within the ``Nalu-Wind/build`` location. You may also build a debug executable by modifying the Nalu-Wind config file to use "Debug". In this case, a ``naluXd`` executable is created.


Build Nalu-Wind with ParaView Catalyst Enabled
**********************************************

If you have built ParaView Catalyst and the Nalu-Wind ParaView Catalyst Adapter, you
can build Nalu-Wind with Catalyst enabled.

In ``Nalu-Wind/build``, find ``do-configNaluCatalyst``. Copy ``do-configNaluCatalyst`` to
a new, non-tracked file:

::

    cp do-configNaluCatalyst do-configNaluCatalystNonTracked
    ./do-configNaluCatalystNonTracked
    make 

The build will create the same executables as a regular Nalu-Wind build, and will also create a  
bash shell script named ``naluXCatalyst``.  Use ``naluXCatalyst`` to run Nalu-Wind
with Catalyst enabled.  It is also possible to run ``naluX`` with Catalyst enabled by
first setting the environment variable:

::

   export CATALYST_ADAPTER_INSTALL_DIR=$nalu_build_dir/install

Nalu-Wind will render images to Catalyst in-situ if it encounters the keyword ``catalyst_file_name``
in the ``output`` section of the Nalu-Wind input deck. The ``catalyst_file_name`` command specifies the
path to a text file containing ParaView Catalyst input deck commands. Consult the ``catalyst.txt`` files
in the following Nalu-Wind regression test directories for examples of the Catalyst input deck command syntax:

::

    ablForcingEdge/
    mixedTetPipe/
    steadyTaylorVortex/

::

    output:
      output_data_base_name: mixedTetPipe.e
      catalyst_file_name: catalyst.txt

When the above regression tests are run, Catalyst is run as part of the regression test. The regression
test checks that the correct number of image output files have been created by the test.

The Nalu-Wind Catalyst integration also supports running Catalyst Python script files exported from the ParaView GUI.
The procedure for exporting Catalyst Python scripts from ParaView is documented in the 
`Catalyst user guide <https://www.paraview.org/in-situ/>`__. To use an exported Catalyst script, insert 
the ``paraview_script_name`` keyword in the ``output`` section of the Nalu-Wind input deck. The argument for
the ``paraview_script_name`` command contains a file path to the exported script. 

::

    output:
      output_data_base_name: mixedTetPipe.e
      paraview_script_name: paraview_exported_catalyst_script.py


