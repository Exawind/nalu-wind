Verifying that the installation of Kynema-UGF works
===================================================

To verify that the build was successful and that your installation of Kynema-UGF works you can run the unit tests and one of the regression test cases.

Unit Tests
----------

Running
~~~~~~~

Create a new folder and change into it (the following command creates various files, and this makes it easier to delete them again)
Then, run the binary :code:`unittestX` contained in your installation of Kynema-UGF.

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

Change to the directory `reg_tests/test_files/ablNeutralEdge` and run `kynema-ugf` on the input there:

::

    cd reg_tests/test_files/ablNeutralEdge
    kynema-ugf -i ablNeutralEdge.yaml

Expected result
~~~~~~~~~~~~~~~

Kynema-UGF will now run (should take a moment).
It should log nothing or at most a few informational lines and exit cleanly (exit code 0).
Various new files should be created.
