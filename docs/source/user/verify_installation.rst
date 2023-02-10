Verifying that the installation of Nalu-Wind works
==================================================

To verify that the build was successful and that your installation of Nalu-Wind works you can run the unit tests and one of the regression test cases.

Unit Tests
----------

Running
~~~~~~~

Create a new folder and change into it (the following command creates various files, and this makes it easier to delete them again)
Then, run the binary :code:`unittestX` contained in your installation of Nalu-Wind.

Expected result
~~~~~~~~~~~~~~~

At the end of the output a test summary should be printed.
Ideally, all tests should have passed with possibly a few being skipped and no failures.
In this case the exit code should be 0.

However, a few failures do not necessarily imply your installation is broken.
It could be caused by various factors.

Regression Test Case
--------------------

Preparation
~~~~~~~~~~~

First download the submodule `reg_tests/mesh`:

::

    git submodule update --init reg_tests/mesh/

Running
~~~~~~~

Change to the directory `reg_tests/test_files/ablNeutralEdge` and run `naluX` on the input there:

::

    cd reg_tests/test_files/ablNeutralEdge
    naluX -i ablNeutralEdge.yaml

Expected result
~~~~~~~~~~~~~~~

Nalu-Wind will now run (should take a moment).
It should log nothing or at most a few informational lines and exit cleanly (exit code 0).
Various new files should be created.
