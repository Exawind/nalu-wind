#=============================================================================
# Nalu Testing
#=============================================================================

# Check if mesh submodule is populated
file(GLOB RESULT ${CMAKE_SOURCE_DIR}/reg_tests/mesh/*)
list(LENGTH RESULT RES_LEN)
if(RES_LEN EQUAL 0)
  message(WARNING
"Testing is enabled, but the mesh submodule is not initialized.\
 You should 'git submodule init' and 'git submodule update'\
 before running the tests.")
endif()

set(NALU_WIND_REFERENCE_GOLDS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_files" CACHE PATH
  "Custom directory for checking gold norms")

if (NOT "${NALU_WIND_REFERENCE_GOLDS_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/test_files")
  if(EXISTS ${NALU_WIND_REFERENCE_GOLDS_DIR})
    set(NALU_WIND_REFERENCE_GOLDS_DIR "${NALU_WIND_REFERENCE_GOLDS_DIR}/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "Reference gold files will be expected here: ${NALU_WIND_REFERENCE_GOLDS_DIR}")
  else()
    message(FATAL_ERROR "Reference gold files path does not exist: ${NALU_WIND_REFERENCE_GOLDS_DIR}")
  endif()
  if("${TEST_ABS_TOL}" STREQUAL "")
    set(TEST_ABS_TOL "1.0e-15")
  endif()
  if("${TEST_REL_TOL}" STREQUAL "")
    set(TEST_REL_TOL "1.0e-12")
  endif()
endif()

if("${TEST_ABS_TOL}" STREQUAL "")
  set(TEST_ABS_TOL "1000.0") # Useless default
endif()

if("${TEST_REL_TOL}" STREQUAL "")
  set(TEST_REL_TOL "1000.0") # Useless default
endif()
message(STATUS "Using test tolerance: abs = ${TEST_ABS_TOL}, rel = ${TEST_REL_TOL}")

if(NALU_WIND_SAVE_GOLDS)
  if("${NALU_WIND_SAVED_GOLDS_DIR}" STREQUAL "")
    message(FATAL_ERROR "To save gold files, NALU_WIND_SAVED_GOLDS_DIR must be set and the directory exist")
  else()
    if(EXISTS ${NALU_WIND_SAVED_GOLDS_DIR})
      set(SAVED_GOLDS_DIR ${NALU_WIND_SAVED_GOLDS_DIR}/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION})
      message(STATUS "Gold files will be saved to: ${SAVED_GOLDS_DIR}")
    else()
      message(FATAL_ERROR "Specified directory for saving gold files does not exist: ${NALU_WIND_SAVED_GOLDS_DIR}")
    endif()
  endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

# Create symlinks to mesh and xml directories
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/mesh
                ${CMAKE_CURRENT_BINARY_DIR}/mesh)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/xml
                ${CMAKE_CURRENT_BINARY_DIR}/xml)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/hypre_settings
                ${CMAKE_CURRENT_BINARY_DIR}/hypre_settings)

# Add tests in list
include(${CMAKE_CURRENT_SOURCE_DIR}/CTestList.cmake)
