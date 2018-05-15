add_sources(
   AssembleOversetSolverConstraintAlgorithm.C
   OversetInfo.C
   OversetManager.C
   OversetManagerSTK.C
   UpdateOversetFringeAlgorithmDriver.C
)

if(ENABLE_TIOGA)
   add_sources(
      OversetManagerTIOGA.C
      TiogaBlock.C
      TiogaSTKIface.C
   )
endif()
