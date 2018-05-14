Building the Documentation
==========================

This document describes how to build Nalu-Wind's documentation.
The documentation is based on the use of Doxygen, Sphinx,
and Doxylink. Therefore we will need to install these tools
as well as some extensions of Sphinx that are utilized.

Install the Tools
-----------------

Install CMake, Doxygen, Sphinx, Doxylink, and the
extensions used. Doxygen uses the ``dot`` application
installed with GraphViz. Sphinx uses a combination
of extensions installed with ``pip install`` as well as some
that come with Nalu-Wind located in the ``_extensions``
directory. Using Homebrew on Mac OS X, 
this would look something like:

::

  brew install cmake
  brew install python
  brew install doxygen
  brew install graphviz
  pip2 install sphinx
  pip2 install sphinxcontrib-bibtex
  pip2 install breathe
  pip2 install sphinx_rtd_theme

On Linux, CMake, Python, Doxygen, and GraphViz could be installed
using your package manager, e.g. ``sudo apt-get install cmake``.

Run CMake Configure
-------------------

In the `Nalu-Wind repository <https://github.com/Exawind/nalu-wind>`__ checkout, 
create your own or use the ``build`` directory that already exists in the repo.
Change to your designated build directory and run CMake with ``-DENABLE_DOCUMENTATION``
on. For example:

::

  cmake -DTrilinos_DIR:PATH=$(spack location -i nalu-trilinos) \
        -DYAML_DIR:PATH=$(spack location -i yaml-cpp) \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DENABLE_DOCUMENTATION:BOOL=ON \
        ..

If all of the main tools are found successfully, CMake should configure with the ability
to build the documentation. If Sphinx or Doxygen aren't found, the configure will skip
the documentation.

Make the Docs
-------------

In your designated build directory, issue the command ``make docs`` which 
should first build the Doxygen documentation and then the Sphinx documentation. 
If this completes successfully, the entry point to
the documentation should be in ``build/docs/html/index.html``.
