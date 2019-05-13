Building Nalu-Wind Semi-Automatically Using Spack
=================================================

Mac OS X or Linux
-----------------

The following describes how to build Nalu-Wind and its dependencies
mostly automatically on your Mac using 
`Spack <https://spack.readthedocs.io/en/latest>`__. 
This can also be used as a template to build Nalu-Wind on any 
Linux system with Spack.

Step 1
~~~~~~

This assumes you have a (Homebrew) installation of GCC installed already 
(we are using GCC 7.3.0). These instructions have been tested on OSX 10.11, MacOS 10.12, and MacOS 10.13.
MacOS 10.12/10.13 will not build CMake or Pkg-Config with GCC because they will pick up 
system header files that have objective C code in them. Therefore we build Nalu-Wind using Spack on MacOS Sierra by
using Homebrew to install ``cmake`` and ``pkg-config`` and defining these 
as external packages in Spack (see 
`packages.yaml <https://github.com/exawind/build-test/blob/master/configs/machines/mac_sierra/packages.yaml>`__).

Step 2
~~~~~~

Checkout the official Spack repo from github (we will checkout into ``${HOME}``):

::

    cd ${HOME} && git clone https://github.com/spack/spack.git

Step 3
~~~~~~

Add Spack shell support to your ``.profile`` or ``.bashrc`` etc, by adding the lines:

::

    export SPACK_ROOT=${HOME}/spack
    source ${SPACK_ROOT}/share/spack/setup-env.sh

Step 4
~~~~~~

Run the `setup-spack.sh <https://github.com/exawind/build-test/blob/master/configs/setup-spack.sh>`__
script from the repo which tries to find out what machine you are on and then copies the corresponding ``*.yaml`` 
configuration files to your Spack installation:

::

    cd ${HOME} && git clone https://github.com/exawind/build-test.git
    cd ${HOME}/build-test/configs && ./setup-spack.sh

Step 5
~~~~~~

Try ``spack info nalu-wind`` to see if Spack works. If it does, check the
compilers you have available by:

::

    machine:~ user$ spack compilers
    ==> Available compilers
    -- clang sierra-x86_64 ------------------------------------------
    clang@9.0.0-apple
    
    -- gcc sierra-x86_64 --------------------------------------------
    gcc@7.3.0  gcc@6.4.0  gcc@5.5.0

Step 6
~~~~~~

Install Nalu-Wind with whatever compiler you prefer (it will default to Apple Clang) merely by
running ``spack install nalu-wind`` or by editing and running the 
``install_nalu_gcc_mac.sh`` script from the `build-test <https://github.com/exawind/build-test>`__ repo:

::

    cd ${HOME}/build-test/install_scripts && ./install_nalu_gcc_mac.sh

That should be it! When using the install script you will see that Spack will install
using the constraints we've specified in ``shared_constraints.sh`` which specifies a much more specific
set of Trilinos options for Nalu-Wind that can shorten the build time.


NREL's Eagle Machine
--------------------

The following describes how to build Nalu-Wind and its dependencies
mostly automatically on NREL's Eagle machine using Spack. This can also be
used as a template to help build Nalu-Wind on any Linux system with Spack.

Step 1
~~~~~~

Login to Eagle, and checkout the ``https://github.com/exawind/build-test.git`` 
repo (we will be cloning into the ${HOME} directory):

::

   cd ${HOME} && git clone https://github.com/exawind/build-test.git

Step 2
~~~~~~

Checkout the official Spack repo from github:

``cd ${HOME} && git clone https://github.com/spack/spack.git``

Step 3
~~~~~~

Configure your environment in the recommended way. You should purge all 
modules and load GCC 7.3.0 in your login script. In the example 
`.bashrc <https://github.com/exawind/build-test/blob/master/configs/machines/eagle/dot_bashrc_eagle.sh>`__
in the repo we also load Python. If you have problems building with Spack on 
Eagle, it is most likely your environment has deviated from this 
recommended one. Even when building with the Intel compiler in Spack, 
this is the recommended environment at login.

::

   module purge
   module load gcc/7.3.0

Also add Spack shell support to your ``.bashrc`` as shown in the example 
`.bashrc <https://github.com/exawind/build-test/blob/master/configs/machines/eagle/dot_bashrc_eagle.sh>`__
in the repo or the following lines:

::

   export SPACK_ROOT=${HOME}/spack
   source ${SPACK_ROOT}/share/spack/setup-env.sh

Log out and log back in or source your ``.bashrc`` to get the Spack 
shell support loaded. Try ``spack info nalu-wind`` to see if Spack works.

Step 4
~~~~~~

Configure Spack for Eagle. This is done by running the
`setup-spack.sh <https://github.com/exawind/build-test/blob/master/configs/setup-spack.sh>`__
script provided which tries finding what machine you're on and copying the corresponding ``*.yaml``
file to your Spack directory:

::

   cd ${HOME}/build-test/configs && ./setup-spack.sh

Step 5
~~~~~~

Try ``spack info nalu-wind`` to see if Spack works.

Step 6
~~~~~~

Note the build scripts and packages.yaml configuration files provided here adhere
to the official versions of the third party libraries 
we test with, and that you may want to adhere to using them as well. Also note that
when you checkout the latest Spack, it also means you will be using the latest packages 
available if you do not set constraints at install time and the newest packages 
may not have been tested to build correctly on NREL machines yet. So specifying
versions of the TPL dependencies in your packages.yaml file for Spack is recommended.

Install Nalu-Wind using a non-GPU login node with the example script  
`install_nalu_eagle.sh <https://github.com/exawind/build-test/blob/master/install_scripts/install_nalu_eagle.sh>`__
or edit the script to use the correct allocation and ``nice ./install_nalu_eagle.sh``.

That's it! Hopefully the ``install_nalu_eagle.sh`` 
script installs the entire set of dependencies and you get a working build 
of Nalu-Wind on Eagle...after several hours of waiting for it to build.

To build with the Intel compiler, note the necessary change listed in the
`install_nalu_eagle.sh <https://github.com/exawind/build-test/blob/master/install_scripts/install_nalu_eagle.sh>`__ 
batch script.

To load Nalu-Wind dependencies (you will need Spack's OpenMPI for Nalu-Wind now) into your path you 
will need to ``spack load openmpi %compiler`` and ``spack load nalu-wind %compiler``, using 
``%gcc`` or ``%intel`` to specify which to load.

Development Build of Nalu-Wind
------------------------------

When building Nalu-Wind with Spack, Spack will cache downloaded archive files such as
``*.tar.gz`` files. However, by default Spack will also erase extracted or
checked out ('staged') source files after it has built a package successfully. 
Therefore if your build succeeds, Spack will have erased the Nalu-Wind source code 
it checked out from Github. 

The recommended way to get a version of Nalu-Wind you can develop in 
is to checkout Nalu-Wind yourself outside of Spack and build this version 
using the dependencies Spack has built for you. To do so, checkout Nalu-Wind:

::

   git clone https://github.com/exawind/nalu-wind.git

Next, create your own directory to build in, or use the existing ``build`` directory in Nalu-Wind to 
run the CMake configuration. When running the CMake configuration, point Nalu-Wind to 
the dependencies by using ``spack location -i <package>``. For example in the 
``build`` directory run:

::

   cmake -DTrilinos_DIR:PATH=$(spack location -i trilinos) \
         -DYAML_DIR:PATH=$(spack location -i yaml-cpp) \
         -DCMAKE_BUILD_TYPE=RELEASE \
         ..
   make

There are also ``do-config`` scripts available for this according to machine under the configs directory `here <https://github.com/exawind/build-test>`__. These scripts may also provide the capability to access and use pre-built dependencies from a set of modules if they are available on the machine. This should allow you to have a build of Nalu-Wind in which you are able to continuosly modify the source code and rebuild.
