target_sources(${kynema_ugf_lib_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetDecoupledAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetPressureAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetSolverConstraintAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/AssembleOversetWallDistAlgorithm.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetConstraintBase.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetInfo.C
   ${CMAKE_CURRENT_SOURCE_DIR}/OversetManager.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UpdateOversetFringeAlgorithmDriver.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ExtOverset.C
   ${CMAKE_CURRENT_SOURCE_DIR}/overset_utils.C
)

if(ENABLE_TIOGA)
   target_sources(${kynema_ugf_lib_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/OversetManagerTIOGA.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaOptions.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaBlock.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaSTKIface.C
      ${CMAKE_CURRENT_SOURCE_DIR}/TiogaRef.C
   )
   if(ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
      # Work around nvcc/cudafe++ ICE (CUDA 12.8 + GCC 12.5) in TiogaOptions.C.
      set_source_files_properties(
         ${CMAKE_CURRENT_SOURCE_DIR}/TiogaOptions.C PROPERTIES COMPILE_OPTIONS
                                                         "--host-only")
   endif()
endif()
