Writing Developer Documentation
===============================

Developer documentation should be written using Doxygen annotations directly in
the source code. This allows the documentation to live with the code essentially
as comments that Doxygen is able to extract automatically into a more human
readable form. Doxygen requires special syntax markers to indicate comments that
should be processed as inline documentation vs. generic comments in the source
code. The `Doxygen manual
<http://www.stack.nl/~dimitri/doxygen/manual/index.html>`_ provides detailed
information on the various markers available to tailor the formatting of
auto-generated documentation. It is recommended that users document the classes
and methods in the header file. A sample header file with specially formatted
comments is shown below. You can :download:`download <dox_example/example.h>` a
copy of the file.


.. literalinclude:: dox_example/example.h
   :language: c++
   :caption: Sample C++ header file showing inline documentation using specially formatted comments.
