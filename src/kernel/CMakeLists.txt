target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/ContinuityInflowElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/EnthalpyTGradBCElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/KernelBuilderLog.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumOpenAdvDiffElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumSymmetryElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumWallFunctionElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarFluxBCElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarFaceFluxBCElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarOpenAdvElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/WallDistElemKernel.C
)

