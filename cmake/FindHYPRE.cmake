# Find HYPRE linear solver library
#
# Set HYPRE_DIR to the base directory where the package is installed
#
# Sets two variables
#   - HYPRE_INCLUDE_DIRS
#   - HYPRE_LIBRARIES
#

find_path(HYPRE_INCLUDE_DIRS
  HYPRE.h
  HINTS ${HYPRE_ROOT} ${HYPRE_DIR} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES include)

find_library(HYPRE_LIBRARIES
  NAMES HYPRE HYPRE
  HINTS ${HYPRE_ROOT} ${HYPRE_DIR} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib)

if (HYPRE_INCLUDE_DIRS)
  file(STRINGS ${HYPRE_INCLUDE_DIRS}/HYPRE_config.h
    _hypre_version_string REGEX "HYPRE_RELEASE_VERSION")
  string(REGEX MATCHALL "[0-9]+" HYPRE_VERSION "${_hypre_version_string}")
  string(REPLACE ";" "." HYPRE_VERSION "${HYPRE_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HYPRE DEFAULT_MSG HYPRE_VERSION HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)
mark_as_advanced(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)

if (HYPRE_FOUND AND NOT TARGET HYPRE)
  add_library(HYPRE UNKNOWN IMPORTED GLOBAL)
  set_target_properties(HYPRE
    PROPERTIES
    IMPORTED_LOCATION "${HYPRE_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
endif()
