target_sources(nalu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/AeroContainer.C
  ${CMAKE_CURRENT_SOURCE_DIR}/CalcLoads.C
)

add_subdirectory(actuator)

if(ENABLE_OPENFAST)
add_subdirectory(fsi)
endif()

add_subdirectory(six_dof)
