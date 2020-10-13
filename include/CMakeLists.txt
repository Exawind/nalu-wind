# Add a custom target with headers for IDEs (e.g., XCode)
file(GLOB_RECURSE nalu_wind_header_list ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
add_custom_target(nalu_wind_headers SOURCES ${nalu_wind_header_list})
