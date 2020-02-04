target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/Actuator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorBulk.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorParsing.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorSearch.C
   )

if(ENABLE_OPENFAST)
   target_sources(nalu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/UtilitiesActuator.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorLineFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/ActuatorDiskFAST.C
   )
endif()