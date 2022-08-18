target_sources(nalu PRIVATE
  # Edge kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/AssembleEdgeKernelAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/AssembleAMSEdgeKernelAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ContinuityEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSSTAMSDiffEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSSTLRAMSDiffEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumKEAMSDiffEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumKOAMSDiffEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/WallDistEdgeSolverAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumEdgePecletAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/StreletsUpwindEdgeAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/AMSMomentumEdgePecletAlg.C
  ${CMAKE_CURRENT_SOURCE_DIR}/VOFAdvectionEdgeAlg.C

  # Face/edge BC kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/ContinuityOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumABLWallFuncEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumABLWallShearStressEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSymmetryEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarOpenEdgeKernel.C
  ${CMAKE_CURRENT_SOURCE_DIR}/ScalarEdgeOpenSolverAlg.C
  )
