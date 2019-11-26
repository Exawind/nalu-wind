target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/ElementCondenser.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ElementDescription.C
   ${CMAKE_CURRENT_SOURCE_DIR}/HexNElementDescription.C
   ${CMAKE_CURRENT_SOURCE_DIR}/LagrangeBasis.C
   ${CMAKE_CURRENT_SOURCE_DIR}/NodeMapMaker.C
   ${CMAKE_CURRENT_SOURCE_DIR}/PromoteElement.C
   ${CMAKE_CURRENT_SOURCE_DIR}/PromotedElementIO.C
   ${CMAKE_CURRENT_SOURCE_DIR}/PromotedPartHelper.C
   ${CMAKE_CURRENT_SOURCE_DIR}/QuadNElementDescription.C
   ${CMAKE_CURRENT_SOURCE_DIR}/QuadratureRule.C
   ${CMAKE_CURRENT_SOURCE_DIR}/TensorProductQuadratureRule.C
)
