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

set(NALU_GOLD_NORMS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
  "Custom directory for checking gold norms")

if (NOT ${NALU_GOLD_NORMS_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(STATUS "Setting custom gold norms path: ${NALU_GOLD_NORMS_DIR}")
  set(TEST_TOLERANCE "1.0e-15")
  set(TEST_REL_TOL "1.0e-12")
endif()

# Set TOLERANCE for testing
if(NOT ${TEST_TOLERANCE} STREQUAL "")
  set(TOLERANCE ${TEST_TOLERANCE}) # User defined
else(NOT ${TEST_TOLERANCE} STREQUAL "")
  # Golds are generated with Linux GCC 8.4.0
  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux"
     AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
     AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 8.0
     AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 9.0)
    set(TOLERANCE "1.0e-15")
  else()
    set(TOLERANCE 0.05) # Otherwise set some useless default
  endif()
endif()

if(NOT ${TEST_REL_TOL} STREQUAL "")
  set(REL_TOLERANCE ${TEST_REL_TOL}) # User defined
else(NOT ${TEST_REL_TOL} STREQUAL "")
  # Golds are generated with Linux GCC 8.4.0
  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux"
      AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
      AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 8.0
      AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 9.0)
    set(REL_TOLERANCE "1.0e-13")
  else()
    set(REL_TOLERANCE "1.0e-7")
  endif()
endif()
message(STATUS "Using test tolerance: abs = ${TOLERANCE}; rel = ${REL_TOLERANCE}")

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
