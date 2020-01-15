target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/ConstantPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/EnthalpyPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/GenericPropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/HDF5TablePropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/IdealGasPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/InverseDualVolumePropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/InversePropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/LinearPropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MaterialPropertyData.C
   ${CMAKE_CURRENT_SOURCE_DIR}/PolynomialPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ReferencePropertyData.C
   ${CMAKE_CURRENT_SOURCE_DIR}/SpecificHeatPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/SutherlandsPropertyEvaluator.C
   ${CMAKE_CURRENT_SOURCE_DIR}/TemperaturePropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ThermalConductivityFromPrandtlPropAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/WaterPropertyEvaluator.C
)
