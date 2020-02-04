target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorParsing.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorBulk.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorNGP.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorSearch.C
)
if(ENABLE_OPENFAST)
   target_sources(${utest_ex_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorDiskFAST.C
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestActuatorFASTGaussian.C
   )
endif()
