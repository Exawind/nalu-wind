target_sources(nalu PRIVATE
  # Edge kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/AssembleEdgeKernelAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/AssembleTAMSEdgeKernelAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ContinuityEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSSTTAMSDiffEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/WallDistEdgeSolverAlg.C

  # Face/edge BC kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/ContinuityOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumABLWallFuncEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSymmetryEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEdgeOpenSolverAlg.C
  )
