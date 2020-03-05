target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameBase.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameMoving.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameReference.C
   ${CMAKE_CURRENT_SOURCE_DIR}/FrameOpenFAST.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshMotionAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MeshTransformationAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionDeformingInteriorKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionRotationKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionScalingKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionTranslationKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MotionWavesKernel.C
)
