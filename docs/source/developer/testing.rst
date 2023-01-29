Testing Nalu-Wind
=================

Nalu-Wind's regression tests and unit tests are run nightly using the GCC and Intel 
compilers against the Trilinos master and development branches on a machine 
at NREL. The results can be seen at the `CDash Nalu-Wind website <http://my.cdash.org/index.php?project=Exawind>`__.


Running Tests Locally
---------------------

The nightly tests are implemented using `CTest <https://cmake.org/cmake/help/v3.7/manual/ctest.1.html>`__ and
these same tests are available to developers to run locally as well. Due to the nature of error propagation of 
calculations in computers, results of regression testing can be difficult to keep consistent. 
Output from Nalu-Wind can vary from established reference data for the regression tests based on the compiler you 
are using, the types of optimizations set, and the versions of the third-party libraries Nalu-Wind 
utilizes, along with the blas/lapack implementation in use. Therefore it may make sense when 
you checkout Nalu-Wind to create your own reference data for the tests for the machine and 
configuration you are using, which is described later in this document. Or you can use a lower tolerance 
when running the tests. At the moment, a single tolerance is chosen in which to use for all the tests. 
The following instructions will describe how to run Nalu-Wind's tests.

Since Nalu-Wind's tests require a large amount of data (meshes), this data is hosted in a separate repository 
from Nalu-Wind. This mesh repo is set as a submodule in the ``reg_tests/mesh`` directory in the main 
Nalu-Wind repository. Submodule repos are not checked out by default, so either use ``git submodule init`` 
and then ``git submodule update`` to clone it in your checkout of Nalu-Wind, or when you first clone Nalu-Wind you can also use 
``git clone --recursive <repo_url>`` to checkout all submodules as well.

Once this submodule is intialized and cloned, you will need to configure Nalu-Wind with testing on.
To configure Nalu-Wind with testing enabled, in Nalu-Wind's existing ``build`` directory, we will run this command:

::

   cmake -DTrilinos_DIR:PATH=`spack location -i nalu-trilinos` \
         -DYAML_DIR:PATH=`spack location -i yaml-cpp` \
         -DENABLE_TESTS:BOOL=ON \
         ..

Note we have chosen to originally build Nalu-Wind with Spack in this case, hence the use 
of ``spack location -i <package>`` to locate our Yaml and Trilinos installations. 
Then we use ``-DENABLE_TESTS:BOOL=ON`` to enable CTest. Once Nalu-Wind is configured, 
you should be able to run the tests by building Nalu-Wind in the ``build`` directory, 
and running ``make test`` or ``ctest``. Looking at ``ctest -h`` will show you many ways 
you can run tests and choose which tests to run.

There are advantages to using CTest, such as being able to run subsets of the tests, or tests 
matching a particular regular expression for example. To do so, in the ``build`` directory, you can run 
``ctest -R femHC`` to run the test matching the ``femHC`` regular expression. Other useful capabilities are 
``ctest --output-on-failure`` to see test outputs when they fail, ``ctest --rerun-failed`` to only run 
the tests that previously failed, ``ctest --print-labels`` to see the test labels, and ``ctest -L unit`` 
to run the tests with label 'unit' for example. All testing related log files and output can be seen in 
``Nalu-Wind/build/Testing/Temporary`` and ``Nalu-Wind/build/reg_tests`` after the test have been run.

To define your own tolerance for tests, at configure time, add ``-DTEST_TOLERANCE=0.0001`` for example 
to the Nalu-Wind CMake configure line.


Updating Reference Data for Your Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running the tests, the norms for each test are the output and they are 'diffed' against 
the 'gold' norms that we have established for each test. To dictate whether or not a test passes, 
we use a chosen tolerance in which we allow the results to deviate from the 'gold' norm.  As stated 
earlier, these 'gold' norms are not able to reflect every configuration of Nalu-Wind, per compiler, optimization, 
TPL versions, blas/lapack version, etc. This tolerance is currently defined in the ``CMakeLists.txt`` 
in Nalu-Wind's ``reg_tests`` directory. This tolerance can also be passed into Nalu-Wind at configure time using 
``-DTEST_TOLERANCE=0.0000001`` for example. To update the 'gold' norms locally to your configuration, merely 
run the tests once, and copy the ``*.norm`` files in the ``build/reg_tests/test_files`` directory 
to the corresponding test location in ``reg_tests/test_files`` while overwriting the current 'gold' norms.

In regards to 'official' gold norms, Linux with GCC 6.4.0, netlib-blas/lapack 3.8.0, and the following 
TPL versions are officially tested:

::

  openmpi@1.10.4
  boost@1.66.0
  cmake@3.9.4
  parallel-netcdf@1.8.0
  yaml-cpp@develop
  hdf5@1.10.1
  netcdf@4.4.1.1
  zlib@1.2.11
  superlu@4.3 


Adding Tests to Nalu-Wind
-------------------------

.. _add-test:

The testing infrastructure is almost completely confined to the ``reg_tests`` directory. To add a test 
to Nalu-Wind, we need to add the test name, and create a test directory to place the input files and gold norms 
for the test. First, the test itself can be added to the list of CTest tests by adding a line to the 
``CTestList.cmake`` file. For a single regression test, provided it is similar to the categories shown at 
the top of the ``CTestList.cmake`` file, it can likely be added with a single line using the test 
name and amount of processes you would like to run the test with and choosing the correct function to use. 
For example:

::

    add_test_r(mytest 6)

After this has been done, in the ``reg_tests/test_files`` directory, you should add a directory corresponding to your 
test name and include the input file, ``mytest.i``, and reference output file ``mytest.norm.gold``. If you are using 
an xml file that doesn't exist in the ``xml`` directory, you will need to commit that as well.

To see commands used when running the tests, see the functions at the top of the ``CTestList.cmake`` file. These 
functions ultimately create ``CTestTestFile.cmake`` files in the CMake build directory at configure time. 
You can see the exact commands used for each test after configure in the 
``build/reg_tests/CTestTestFile.cmake`` file.

Note if your test doesn't conform to an existing archetype, a new function in ``CTestList.cmake`` may need to be 
created. Also, if you are using a mesh file that doesn't exist in the mesh repo, you will need to add it, and 
update the submodule in the Nalu-Wind main repo to use the latest commit of the mesh submodule repo.


.. _ref-testing-cdash:

Adding Testing Machines to CDash
--------------------------------

To add a testing machine that will post results to CDash first means that you should have all software 
dependencies satisified for Nalu-Wind. Next the script located at  
`CTestNightlyScript.cmake <https://github.com/Exawind/nalu-wind/blob/master/reg_tests/CTestNightlyScript.cmake>`__ 
can be run for example as:

::

   ctest \
     -DNIGHTLY_DIR=${NALU_WIND_TESTING_DIR} \
     -DYAML_DIR=${YAML_INSTALL_DIR} \
     -DTRILINOS_DIR=${TRILINOS_INSTALL_DIR} \
     -DHOST_NAME=machine.domain.com \
     -DEXTRA_BUILD_NAME=Linux-gcc-whatever \
     -VV -S ${NALU_WIND_DIR}/reg_tests/CTestNightlyScript.cmake

In this case ``${NALU_WIND_TESTING_DIR}`` is one directory above where the Nalu-Wind repo has been checked out. 
This runs CTest in scripting mode with verbosity on and it will update the Nalu-Wind repo with the latest 
revisions, configure, build, test, and finally submit results to the CDash site. Since CTest does 
the building, it needs to know the locations of Yaml and Trilinos. For examples of nightly testing, 
refer to the testing scripts currently being run 
`here <https://github.com/Exawind/build-test/tree/master/test-scripts>`__.
