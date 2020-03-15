target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/Actuator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulk.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsing.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorSearch.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFunctors.C
   )

if(ENABLE_OPENFAST)
   target_sources(nalu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/UtilitiesActuator.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorLineFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorDiskFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulkFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsingFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFunctorsFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorLineFASTNgp.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulkDiskFAST.C
   )
endif()