Simulation -- Nalu Top-level Interface
======================================

.. doxygenclass:: sierra::nalu::Simulation

Realms
------

Realm is a Nalu abstraction of a set of equations that are solved on a
computational domain, reresented by an Exodus-II mesh. A simulation can contain
multiple Realms and that can interact via :class:`sierra::nalu::Transfer`
instance. :class:`~sierra::nalu::InputOutputRealm` is a special type of Realm
that exists solely to provide data (input) or extract a subset of data from
another :class:`~sierra::nalu::Realm`.


.. doxygenclass:: sierra::nalu::Realm
   :members:

.. doxygenclass:: sierra::nalu::InputOutputRealm
   :members:

.. doxygenclass:: sierra::nalu::Realms
   :members:

Time Integration
----------------

.. doxygenclass:: sierra::nalu::TimeIntegrator
   :members:

Linear Solver Interface
-----------------------

.. doxygenclass:: sierra::nalu::LinearSystem
   :members:

.. doxygenclass:: sierra::nalu::LinearSolver
   :members:

.. doxygenclass:: sierra::nalu::TpetraLinearSystem
   :members:

Transfers
---------

.. doxygenclass:: sierra::nalu::Transfer
   :members:

.. doxygenclass:: sierra::nalu::Transfers
   :members:
