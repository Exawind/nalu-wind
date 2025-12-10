Running Nalu-Wind
=================

This section describes the general process of setting up and executing Nalu-Wind,
understanding the various input file options available to the user, and how to
extract results and analyze them. For the simplest case, Nalu-Wind requires the user
to provide a YAML input file with the options that control the run along with a
computational mesh in Exodus-II format. More complex setups might require
additional files:

  - Trilinos MueLu preconditioner configuration in XML format
  - ParaView Cataylst input file for in-situ visualizations
  - Additional Exodus-II mesh files for solving different physics equation sets
    on different meshes, or for solution transfer to an input/output mesh.

.. toctree::
   :maxdepth: 4

   nalu_run/nalu_mesh
   nalu_run/nalux
   nalu_run/nalu_inp
   nalu_run/McAlisterLessonsLearned
