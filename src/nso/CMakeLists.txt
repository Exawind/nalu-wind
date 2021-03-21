target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOKeElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOSijElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOKeElemKernel.C
)
