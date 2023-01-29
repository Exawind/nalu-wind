Development Infrastructure
==========================

This document describes the development infrastructure used by the nalu-wind project.

Docker Container
----------------

The docker container `ecpe4s/exawind-snapshot <https://hub.docker.com/r/ecpe4s/exawind-snapshot>`_
is build by an external collaborator in the E4S project.
Its definition is at https://gitlab.e4s.io/uo-public/exawind-snapshot.

It is based on an Ubuntu with GCC on which spack-manager is used to install exawind+hypre+openfast.

It is build and pushed to Dockerhub daily.

Continuous Integration
----------------------

There are currently two different systems the continuously check the code of this project.

GitHub Actions
^^^^^^^^^^^^^^

The `github actions workflow <https://github.com/Exawind/nalu-wind/blob/master/.github/workflows/ci.yml>`_
runs on every pull request towards master as well as all commits on master.

It does the following things:

* Style Check of the CPP code using clang-format with clang version 13
* Run the unit tests

  * Uses the docker container to allow to reuse the third party libraries
  * Rebuilds nalu-wind with the source from the commit on which the ci is run
  * Run all test with `unit` in the label using ctest

Daily Regression Tests
^^^^^^^^^^^^^^^^^^^^^^

Additionally, the project is tested more extensively on some machines at NREL.
The results are collected at `CDash <https://my.cdash.org/index.php?project=Exawind>`_.
For more info see :ref:`ref-testing-cdash`.
