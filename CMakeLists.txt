############################ BASE ######################################
cmake_minimum_required (VERSION 3.10.0)
project(Nalu CXX C Fortran)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

########################## OPTIONS #####################################
option(ENABLE_CUDA "Enable build targeting GPU" OFF)
option(ENABLE_TESTS "Enable regression testing." OFF)
option(ENABLE_DOCUMENTATION "Build documentation." OFF)
option(ENABLE_SPHINX_API_DOCS "Link Doxygen API docs to Sphinx" OFF)
option(ENABLE_WIND_UTILS "Build wind utils along with Nalu-Wind" OFF)
option(ENABLE_FFTW "Use the FFTW library to support ABLTopBC" OFF)
option(ENABLE_HYPRE "Use HYPRE Solver library" OFF)
option(ENABLE_OPENFAST
       "Use OPENFAST tpl to get actuator line positions and forces" OFF)
option(ENABLE_PARAVIEW_CATALYST
      "Enable ParaView Catalyst. Requires external installation of Trilinos Catalyst IOSS adapter."
       OFF)
option(ENABLE_TIOGA "Use TIOGA TPL to perform overset connectivity" OFF)
option(ENABLE_ALL_WARNINGS "Show most warnings for most compilers" ON)
option(ENABLE_WERROR "Warnings are errors" OFF)
option(ENABLE_OPENMP "Enable OpenMP flags" ON)

set(CMAKE_CXX_STANDARD 11)       # Set nalu-wind C++11 standard
set(CMAKE_CXX_EXTENSIONS OFF)    # Do not enable GNU extensions
set(CMAKE_CXX_STANDARD_REQUIRED) # Force error if C++11 standard is not supported

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_Fortran_INCLUDE_PATH})

if(ENABLE_OPENMP)
  find_package(OpenMP)
endif()

########################## TRILINOS ####################################
set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)
if(Trilinos_FOUND)
  message("-- Found Trilinos = ${Trilinos_DIR}")
endif()
include_directories(SYSTEM ${Trilinos_INCLUDE_DIRS})
include_directories(SYSTEM ${Trilinos_TPL_INCLUDE_DIRS})

# Build Nalu as shared libraries if that is how Trilinos was built
if(Trilinos_BUILD_SHARED_LIBS)
  set(Nalu_LIBRARY_TYPE SHARED)
  set(BUILD_SHARED_LIBS ON)
else(Trilinos_BUILD_SHARED_LIBS)
  set(Nalu_LIBRARY_TYPE STATIC)
endif(Trilinos_BUILD_SHARED_LIBS)

############################ YAML ######################################
set(CMAKE_PREFIX_PATH ${YAML_DIR} ${CMAKE_PREFIX_PATH})
find_package(YAML-CPP QUIET)
if(YAML-CPP_FOUND)
  # YAML master branch is used
  include_directories(SYSTEM ${YAML_CPP_INCLUDE_DIR})
else()
  # YAML 0.5.3 is used
  find_library(YAML_CPP_LIBRARIES NAMES yaml-cpp PATHS ${YAML_DIR}/lib)
  find_path(YAML_CPP_INCLUDE_DIR yaml.h PATHS ${YAML_DIR}/include/yaml-cpp)
  if((DEFINED YAML_CPP_LIBRARIES) AND (DEFINED YAML_CPP_INCLUDE_DIR))
    include_directories(SYSTEM ${YAML_CPP_INCLUDE_DIR}/..)
    set(YAML-CPP_FOUND TRUE)
  endif()
endif()
if(YAML-CPP_FOUND)
  message("-- Found YAML-CPP = ${YAML_DIR}")
else()
  message(FATAL_ERROR "YAML-CPP NOT FOUND")
endif()

if(ENABLE_CUDA)
    add_definitions(-DUSE_STK_SIMD_NONE)
endif()

############################ FFTW ######################################
if(ENABLE_FFTW)
  set(CMAKE_PREFIX_PATH ${FFTW_DIR} ${CMAKE_PREFIX_PATH})
  find_package(FFTW QUIET REQUIRED)
  if(FFTW_FOUND)
    message("-- Found FFTW = ${FFTW_DIR}")
  endif()
  include_directories(${FFTW_INCLUDE_DIRS})
  add_definitions("-DNALU_USES_FFTW")
endif(ENABLE_FFTW)

############################ HYPRE #####################################
if(ENABLE_HYPRE)
  set(CMAKE_PREFIX_PATH ${HYPRE_DIR} ${CMAKE_PREFIX_PATH})
  find_package(HYPRE QUIET REQUIRED)
  if(HYPRE_FOUND)
    message("-- Found HYPRE = ${HYPRE_DIR}")
  endif()
  include_directories(SYSTEM ${HYPRE_INCLUDE_DIRS})
  add_definitions("-DNALU_USES_HYPRE")
endif(ENABLE_HYPRE)

########################## OPENFAST ####################################
if(ENABLE_OPENFAST)
  set(CMAKE_PREFIX_PATH ${OpenFAST_DIR} ${CMAKE_PREFIX_PATH})
  find_package(OpenFAST REQUIRED)
  if(OpenFAST_FOUND)
    message("-- Found OpenFAST = ${OpenFAST_DIR}")
  endif()
  include_directories(${OpenFAST_INCLUDE_DIRS})
  add_definitions("-DNALU_USES_OPENFAST")
endif(ENABLE_OPENFAST)

############################ TIOGA #####################################
if(ENABLE_TIOGA)
  set(CMAKE_PREFIX_PATH ${TIOGA_DIR} ${CMAKE_PREFIX_PATH})
  find_package(TIOGA REQUIRED)
  if(TIOGA_FOUND)
    message("-- Found TIOGA = ${TIOGA_DIR}")
  endif()
  include_directories(${TIOGA_INCLUDE_DIRS})
  add_definitions("-DNALU_USES_TIOGA")
endif()

########################### NALU #####################################
# Add any extra flags based on compiler and/or OS
message("-- CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message("-- CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(EXTRA_CXX_FLAGS "")
  set(EXTRA_Fortran_FLAGS "")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(EXTRA_CXX_FLAGS "")
  set(EXTRA_Fortran_FLAGS "")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(EXTRA_CXX_FLAGS "")
  set(EXTRA_Fortran_FLAGS "")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(EXTRA_CXX_FLAGS "-restrict")
  set(EXTRA_Fortran_FLAGS "")
endif()

# Logic for showing and handling warnings
if(ENABLE_ALL_WARNINGS)
  # GCC, Clang, and Intel seem to accept these
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      # ifort doesn't like -Wall
      set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall")
    else()
      # Intel always reports some diagnostics we don't necessarily care about
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable:11074,11076")
    endif()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
      # Avoid notes about -faligned-new with GCC > 7
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
    endif()
endif()
# Make warnings errors
if(ENABLE_WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Werror")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
if (ENABLE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")
endif()
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${EXTRA_Fortran_FLAGS}")

if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
endif()

message("-- CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("-- CMAKE_Fortran_FLAGS = ${CMAKE_Fortran_FLAGS}")

if(ENABLE_HYPRE)
  include(CheckSymbolExists)
  set(CMAKE_REQUIRED_INCLUDES "${HYPRE_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_LIBRARIES "${HYPRE_LIBRARIES}")
  check_symbol_exists(
    HYPRE_BIGINT "${HYPRE_INCLUDE_DIRS}/HYPRE_config.h" NALU_HYPRE_BIGINT)
  check_symbol_exists(
    hypre_KRYLOV_COGMRES_HEADER "${HYPRE_INCLUDE_DIRS}/krylov.h" NALU_HYPRE_COGMRES)
  if(NOT NALU_HYPRE_BIGINT)
    message(WARNING
      "HYPRE does not enable 64-bit integer support; will fail on large problems!")
  endif()
  if (NALU_HYPRE_COGMRES)
    add_definitions(-DHYPRE_COGMRES)
  else()
    message("-- Disabling HYPRE CO-GMRES interface")
  endif()
endif()

########################### TAG VERSIONS #####################################
include(GetGitRevisionDescription)
get_git_head_revision(NALU_GITREFSPEC NALU_GIT_COMMIT_SHA)
if (NALU_GIT_COMMIT_SHA)
  git_describe(NALU_VERSION_TAG "--tags")
  git_local_changes(NALU_REPO_DIRTY)
  option(NALU_HAVE_GIT_INFO "Git version tagging for Nalu" ON)
  if (${NALU_VERSION_TAG} MATCHES ".*-NOTFOUND")
    set(NALU_VERSION_TAG "v1.2.0")
  endif()
endif()

find_file(TRILINOS_HAVE_GIT_INFO
  "TrilinosRepoVersion.txt"
  PATHS "${Trilinos_DIR}/../../../"
  DOC "Check if Trilinos Git version info exists"
  NO_DEFAULT_PATH)
if (NOT ${TRILINOS_HAVE_GIT_INFO} MATCHES ".*-NOTFOUND")
  file(STRINGS "${Trilinos_DIR}/../../../TrilinosRepoVersion.txt"
    TRILINOS_REPO_VERSION_TXT LIMIT_INPUT 1024)
  list(GET TRILINOS_REPO_VERSION_TXT 1 TRILINOS_REPO_COMMIT_STR)
  string(REGEX MATCH "^[a-z0-9]+" TRILINOS_GIT_COMMIT_SHA ${TRILINOS_REPO_COMMIT_STR})
  set(TRILINOS_VERSION_TAG "${Trilinos_VERSION}-g${TRILINOS_GIT_COMMIT_SHA}")
else()
  set(TRILINOS_VERSION_TAG "${Trilinos_VERSION}")
endif()
string(TIMESTAMP NALU_VERSION_TIMESTAMP "%Y-%m-%d %H:%M:%S (UTC)" UTC)
configure_file("${CMAKE_SOURCE_DIR}/cmake/NaluVersionInfo.h.in"
  "${CMAKE_BINARY_DIR}/include/NaluVersionInfo.h" @ONLY)
#### END TAG VERSIONS

# Add all source files from source subdirectories
function(add_sources target_list)
   foreach(_source IN ITEMS ${ARGN})
       if (IS_ABSOLUTE "${_source}")
           set(_source_abs "${_source}")
       else()
           get_filename_component(_source_abs "${_source}" ABSOLUTE)
       endif()
       set_property(GLOBAL APPEND PROPERTY "${target_list}" "${_source_abs}")
   endforeach()
endfunction(add_sources)
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(unit_tests)
get_property(NALU_SOURCES GLOBAL PROPERTY GlobalSourceList)
get_property(NALU_HEADERS GLOBAL PROPERTY GlobalHeaderList)
get_property(NALU_UNIT_SOURCES GLOBAL PROPERTY GlobalUnitSourceList)
get_property(NALU_UNIT_HEADERS GLOBAL PROPERTY GlobalUnitHeaderList)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
add_library(nalu ${NALU_SOURCES})
target_link_libraries(nalu ${Trilinos_LIBRARIES})
target_link_libraries(nalu ${YAML_CPP_LIBRARIES})
target_link_libraries(nalu ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

if(ENABLE_FFTW)
  target_link_libraries(nalu ${FFTW_LIBRARIES})
endif()

if(ENABLE_HYPRE)
  target_link_libraries(nalu ${HYPRE_LIBRARIES})
endif()

if(ENABLE_OPENFAST)
  target_link_libraries(nalu ${OpenFAST_LIBRARIES} ${OpenFAST_CPP_LIBRARIES})
endif()

if(ENABLE_TIOGA)
  target_link_libraries(nalu ${TIOGA_LIBRARIES})
endif()

message("-- CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

set(nalu_ex_name "naluX")
add_executable(${nalu_ex_name} nalu.C)
target_link_libraries(${nalu_ex_name} nalu)
if(MPI_COMPILE_FLAGS)
  set_target_properties(${nalu_ex_name} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif(MPI_COMPILE_FLAGS)
if(MPI_LINK_FLAGS)
  set_target_properties(${nalu_ex_name} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif(MPI_LINK_FLAGS)

set(utest_ex_name "unittestX")
add_executable(${utest_ex_name} unit_tests.C ${NALU_UNIT_SOURCES})
target_link_libraries(${utest_ex_name} nalu)
target_include_directories(${utest_ex_name} PUBLIC "${CMAKE_SOURCE_DIR}/unit_tests")
if(MPI_COMPILE_FLAGS)
  set_target_properties(${utest_ex_name} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif(MPI_COMPILE_FLAGS)
if(MPI_LINK_FLAGS)
  set_target_properties(${utest_ex_name} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif(MPI_LINK_FLAGS)

set(nalu_ex_catalyst_name "naluXCatalyst")
if(ENABLE_PARAVIEW_CATALYST)
   set(PARAVIEW_CATALYST_INSTALL_PATH
       ""
       CACHE
       PATH
       "Path to external installation of Trilinos Catalyst IOSS plugin.")
   configure_file(cmake/naluXCatalyst.in ${nalu_ex_catalyst_name} @ONLY)
endif()

install(TARGETS ${utest_ex_name} ${nalu_ex_name} nalu
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
if(ENABLE_PARAVIEW_CATALYST)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/naluXCatalyst DESTINATION bin)
endif()

include(CMakePackageConfigHelpers)

if(ENABLE_DOCUMENTATION)
   add_subdirectory(docs)
endif()

if(ENABLE_TESTS)
   enable_testing()
   include(CTest)
   add_subdirectory(reg_tests)
endif()

if (ENABLE_WIND_UTILS)
  if (EXISTS ${CMAKE_SOURCE_DIR}/wind-utils/CMakeLists.txt)
    add_subdirectory(wind-utils)
  else()
    message(WARNING
      "ENABLE_WIND_UTILS is ON, but wind-utils submodule has not been initialized.\
       You should execute 'git submodule init && git submodule update' \
       or use '-DENABLE_WIND_UTILS=OFF' to turn off this warning. \
       DISABLING wind-utils compilation.
      ")
  endif()
endif()

# Note if you are interested in using Xcode for development,
# refer to instructions in https://github.com/Exawind/nalu-wind/pull/22
