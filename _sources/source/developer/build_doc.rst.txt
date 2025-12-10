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

Build the Docs
--------------

In the `Nalu-Wind repository <https://github.com/Exawind/nalu-wind>`__ checkout, execute:

::

  sphinx-build -M html ./docs/sphinx ./build_docs/manual -W --keep-going -n
  doxygen ./docs/doxygen/Doxyfile

If all of the main tools are found successfully, the command will
complete successfully and the entry point to the documentation should
be in ``build_docs/manual/html/index.html`` for the manual and
``build_docs/doxygen/html/index.html`` for the source code.
