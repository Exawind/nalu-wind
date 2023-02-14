target_sources(${utest_ex_name} PRIVATE
  # Edge interior kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestContinuityAdvEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMomentumAdvDiffEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestWallDistEdgeSolver.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestStreletsUpwindEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestVOFAdvectionEdge.C

  # Face/elem edge BC kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestContinuityOpenEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMomentumABLWallFuncEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMomentumOpenEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMomentumSymmetryEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestScalarOpenEdge.C
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMomentumSSTAMSDiffEdge.C
  )

if(ENABLE_TRILINOS_SOLVERS)
  target_sources(${utest_ex_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestScalarAdvDiffEdge.C
  )
endif()
