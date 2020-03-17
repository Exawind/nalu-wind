target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorParsing.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorBulk.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorNGP.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorSearch.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorFunctors.C
)
if(ENABLE_OPENFAST)
   target_sources(${utest_ex_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorFASTUtilities.C
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorFASTGaussian.C
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorParsingFAST.C
   	  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorFunctorsFAST.C
   	  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorBulkDiskFAST.C
   )
endif()
