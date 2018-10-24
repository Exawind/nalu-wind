# Find FFTw library
#
# Set FFTW_DIR to the root directory where the package is installed
#
# Sets two variables
#   - FFTW_INCLUDE_DIRS
#   - FFTW_LIBRARIES
#

find_path(FFTW_INCLUDE_DIRS
  fftw3.h
  HINTS ${FFTW_DIR} ${CMAKE_INSTALL_PREFIX}
  PATHS_SUFFIXES include)

find_library(FFTW_LIBRARIES
  NAMES fftw3
  HINTS ${FFTW_DIR} ${CMAKE_INSTALL_PREFIX}
  PATHS_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDE_DIRS)
mark_as_advanced(FFTW_INCLUDE_DIRS FFTW_LIBRARIES)
