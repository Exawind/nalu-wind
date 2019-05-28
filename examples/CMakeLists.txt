# Copy files from this directory to binary directory
file(GLOB EXAMPLE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*")
file(COPY ${EXAMPLE_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

add_subdirectory(turbine_uniform_inflow)

# Add examples target which builds Nalu and DISCON
add_custom_target(examples DEPENDS ${nalu_ex_name} DISCON)
