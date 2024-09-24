Nalu-Wind - Theory Manual
=========================

Nalu-Wind represents a generalized unstructured, massively parallel, variable density turbulent flow capability designed for energy applications. This code base began as an effort to prototype Sierra Toolkit, :cite:`Edwards:2010`, usage along with direct parallel matrix assembly to the Trilinos, :cite:`Heroux:2003`, Epetra and Tpetra data structure. However, the simulation tool has evolved as a tool to support a variety of research projects germane to the energy sector including wind aerodynamic prediction and traditional gas-phase combustion applications.


.. toctree::
   :numbered:
   :maxdepth: 2
   
   lowMachNumberDerivation.rst
   supportedEquationSet.rst
   discretizationApproach.rst
   advectionStabilization.rst
   pressureStabilization.rst
   rteStabilization.rst
   nso.rst
   turbulenceModeling.rst
   boundaryConditions.rst
   overset.rst
   propertyEvaluation.rst
   couplingApproach.rst
   timeDiscretization.rst
   multiPhysics.rst
   windEnergy.rst
   topologicalSupport.rst
   adaptivity.rst
   codeAbstractions.rst

