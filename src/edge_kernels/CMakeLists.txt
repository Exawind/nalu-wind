add_sources(GlobalSourceList
  # Edge kernels
  ContinuityEdgeSolverAlg.C
  MomentumEdgeSolverAlg.C
  ScalarEdgeSolverAlg.C
  WallDistEdgeSolverAlg.C

  # Face/edge BC kernels
  ContinuityOpenEdgeKernel.C
  MomentumABLWallFuncEdgeKernel.C
  MomentumOpenEdgeKernel.C
  MomentumSymmetryEdgeKernel.C
  ScalarEdgeOpenSolverAlg.C
  )
