target_sources(nalu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/AeroContainer.C
)

add_subdirectory(actuator)

if(ENABLE_OPENFAST)
add_subdirectory(fsi)
endif()

