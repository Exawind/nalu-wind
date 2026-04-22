#Add a custom target with headers for IDEs(e.g., XCode)
file(GLOB_RECURSE kynema_ugf_wind_header_list ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
  add_custom_target(kynema-ugf_wind_headers SOURCES ${kynema_ugf_wind_header_list})
