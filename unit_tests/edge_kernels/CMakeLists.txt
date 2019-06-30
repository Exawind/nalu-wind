add_sources(GlobalUnitSourceList
  # Edge interior kernels
  UnitTestContinuityAdvEdge.C
  UnitTestMomentumAdvDiffEdge.C
  UnitTestScalarAdvDiffEdge.C
  UnitTestWallDistEdgeSolver.C

  # Face/elem edge BC kernels
  UnitTestMomentumABLWallFuncEdge.C
  UnitTestMomentumSymmetryEdge.C
  )
