# Add directory in regression tests which builds the DISCON library
if(NOT ENABLE_TESTS)
  add_subdirectory(${CMAKE_SOURCE_DIR}/reg_tests/test_files/nrel5MWactuatorLine
                   ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine)
endif()

# Create symlinks to nrel5mw.fst from test directory
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
  ${CMAKE_CURRENT_BINARY_DIR}/nrel5mw.fst)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
  ${CMAKE_CURRENT_BINARY_DIR}/nrel5mw2.fst)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_SOURCE_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5MWactuatorLine.i
  ${CMAKE_CURRENT_BINARY_DIR}/template_input_files/nrel5MWactuatorLine.i)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_SOURCE_DIR}/reg_tests/xml/milestone.xml
  ${CMAKE_CURRENT_BINARY_DIR}/milestone.xml)
