target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetDecoupledAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetPressureAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetSolverConstraintAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetWallDistAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetConstraintBase.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetInfo.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetManager.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UpdateOversetFringeAlgorithmDriver.C
)

if(ENABLE_TIOGA)
   target_sources(nalu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/OversetManagerTIOGA.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaOptions.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaBlock.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaSTKIface.C
   )
endif()
