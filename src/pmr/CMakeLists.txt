target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleRadTransEdgeSolverAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleRadTransEdgeUpwindSolverAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleRadTransWallSolverAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransAbsorptionBlackBodyElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransAdvectionSUCVElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransBlackBodyNodeSuppAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransIsoScatteringNodeSuppAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransIsotropicScatteringElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadTransWallElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/RadiativeTransportEquationSystem.C
)
