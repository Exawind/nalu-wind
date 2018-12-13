add_sources(GlobalSourceList
   AssembleOversetPressureAlgorithm.C
   AssembleOversetSolverConstraintAlgorithm.C
   AssembleOversetWallDistAlgorithm.C
   OversetConstraintBase.C
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
