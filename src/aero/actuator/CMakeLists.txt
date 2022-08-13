target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorModel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorExecutor.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulk.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBladeDistributor.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsing.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorSearch.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFunctors.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFLLC.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulkSimple.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFunctorsSimple.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsingSimple.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorExecutorsSimpleNgp.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UtilitiesActuator.C
   )

if(ENABLE_OPENFAST)
   target_sources(nalu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulkFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsingFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFunctorsFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorExecutorsFASTNgp.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulkDiskFAST.C
   )
endif()
