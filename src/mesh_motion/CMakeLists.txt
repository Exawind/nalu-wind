target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleMeshDisplacementElemSolverAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssemblePressureForceBCSolverAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacementEquationSystem.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacementMassBackwardEulerNodeSuppAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameBase.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameInertial.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameNonInertial.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacementEquationSystem.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacementMassBackwardEulerNodeSuppAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshMotionAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionBase.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionPulsatingSphere.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionRotation.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionScaling.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionTranslation.C
)
