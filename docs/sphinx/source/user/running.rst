Running Kynema-UGF
=================

This section describes the general process of setting up and executing Kynema-UGF,
understanding the various input file options available to the user, and how to
extract results and analyze them. For the simplest case, Kynema-UGF requires the user
to provide a YAML input file with the options that control the run along with a
computational mesh in Exodus-II format. More complex setups might require
additional files:

  - Trilinos MueLu preconditioner configuration in XML format
  - ParaView Cataylst input file for in-situ visualizations
  - Additional Exodus-II mesh files for solving different physics equation sets
    on different meshes, or for solution transfer to an input/output mesh.

.. toctree::
   :maxdepth: 4

   kynema-ugf_run/kynema-ugf_mesh
   kynema-ugf_run/kynema-ugfx
   kynema-ugf_run/kynema-ugf_inp
   kynema-ugf_run/McAlisterLessonsLearned
