Verifying that the installation of Nalu-Wind works
==================================================

To verify that the build was successful and that your installation of Nalu-Wind works you can run one of the regression test cases.

Preparation
-----------

First download the submodule `reg_tests/mesh`:

::

    git submodule update --init reg_tests/mesh/

Running
-------

Change to the directory `reg_tests/test_files/ablNeutralEdge` and run `naluX` on the input there:

::

    cd reg_tests/test_files/ablNeutralEdge
    naluX -i ablNeutralEdge.yaml

Expected result
---------------

Nalu-Wind will now run (should take a moment).
It should log nothing or at most a few informational lines and exit cleanly (exit code 0).
Various new files should be created.
