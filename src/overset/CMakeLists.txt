add_sources(GlobalSourceList
   AssembleOversetSolverConstraintAlgorithm.C
   OversetInfo.C
   OversetManager.C
   OversetManagerSTK.C
   UpdateOversetFringeAlgorithmDriver.C
)

if(ENABLE_TIOGA)
   add_sources(GlobalSourceList
      OversetManagerTIOGA.C
      TiogaBlock.C
      TiogaSTKIface.C
   )
endif()
