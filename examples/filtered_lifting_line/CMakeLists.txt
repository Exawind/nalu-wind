# Add directory in regression tests which builds the DISCON library
if(NOT ENABLE_TESTS)
  add_subdirectory(${CMAKE_SOURCE_DIR}/reg_tests/test_files/nrel5MWactuatorLine
                   ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common_files
                 ${CMAKE_CURRENT_BINARY_DIR}/common_files)
