target_sources(${utest_ex_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestNgpAlgUtils.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestCFLReAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestNodalGradAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestDiffFluxCoeffAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestEffSSTDiffFluxCoeffAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestEnthalpyDiffFluxCoeffAlg.C 
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMdotAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestTurbViscKsgsAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestTurbViscSSTAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestGeometryAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestSDRWallAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestNodalGradPOpenBoundary.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestSSTMaxLengthScaleAlg.C
  )
