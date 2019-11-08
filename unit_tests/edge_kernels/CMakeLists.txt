target_sources(${utest_ex_name} PRIVATE
  # Edge interior kernels
  UnitTestContinuityAdvEdge.C
  UnitTestMomentumAdvDiffEdge.C
  UnitTestScalarAdvDiffEdge.C
  UnitTestWallDistEdgeSolver.C

  # Face/elem edge BC kernels
  UnitTestContinuityOpenEdge.C
  UnitTestMomentumABLWallFuncEdge.C
  UnitTestMomentumOpenEdge.C
  UnitTestMomentumSymmetryEdge.C
  UnitTestScalarOpenEdge.C
  )
