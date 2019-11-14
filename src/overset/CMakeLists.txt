target_sources(nalu PRIVATE
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
   target_sources(nalu PRIVATE
      OversetManagerTIOGA.C
      TiogaOptions.C
      TiogaBlock.C
      TiogaSTKIface.C
   )
endif()
