target_sources(${kynema_ugf_lib_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/AeroContainer.C
  ${CMAKE_CURRENT_SOURCE_DIR}/CalcLoads.C
)

if(ENABLE_KYNEMA_FMB_SIXDOF)
target_sources(${kynema_ugf_lib_name} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/FMBContainer.C
)
endif()

add_subdirectory(actuator)

if(ENABLE_OPENFAST)
add_subdirectory(fsi)
endif()

if(ENABLE_KYNEMA_FMB_SIXDOF)
add_subdirectory(fmb)
endif()