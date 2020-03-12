############################ BASE ######################################
cmake_minimum_required (VERSION 3.10.0 FATAL_ERROR)
project(Nalu CXX Fortran)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

########################## OPTIONS #####################################
option(ENABLE_CUDA "Enable build targeting GPU" OFF)
option(ENABLE_TESTS "Enable regression testing." OFF)
option(ENABLE_EXAMPLES "Enable examples." OFF)
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
option(ENABLE_OPENMP "Enable OpenMP flags" OFF)

set(CMAKE_CXX_STANDARD 11)       # Set nalu-wind C++11 standard
set(CMAKE_CXX_EXTENSIONS OFF)    # Do not enable GNU extensions
set(CMAKE_CXX_STANDARD_REQUIRED) # Force error if C++11 standard is not supported

# Create targets
set(nalu_ex_name "naluX")
set(utest_ex_name "unittestX")
add_library(nalu "")
add_executable(${nalu_ex_name} ${CMAKE_CURRENT_SOURCE_DIR}/nalu.C)
add_executable(${utest_ex_name} ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.C)

########################## MPI ####################################
find_package(MPI REQUIRED)
target_link_libraries(nalu PUBLIC $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>)
target_link_libraries(nalu PUBLIC $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>)

########################## TRILINOS ####################################
set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET REQUIRED)
if(Trilinos_FOUND)
  message(STATUS "Found Trilinos = ${Trilinos_LIBRARY_DIRS}")
  target_link_libraries(nalu PUBLIC ${Trilinos_LIBRARIES})
  target_include_directories(nalu SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
  target_include_directories(nalu SYSTEM PUBLIC ${Trilinos_TPL_INCLUDE_DIRS})
  if(Trilinos_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
  endif()
endif()

############################ BOOST #####################################
set(CMAKE_PREFIX_PATH ${Boost_DIR} ${CMAKE_PREFIX_PATH})
find_package(Boost REQUIRED QUIET COMPONENTS filesystem)
if(Boost_FOUND)
  message(STATUS "Found Boost = ${Boost_LIBRARY_DIRS}")
  target_link_libraries(nalu PUBLIC ${Boost_LIBRARIES})
endif()

############################ YAML ######################################
set(CMAKE_PREFIX_PATH ${YAML_DIR} ${CMAKE_PREFIX_PATH})
find_package(YAML-CPP QUIET)
if(YAML-CPP_FOUND)
  # YAML 0.6.2 or master branch is used
  target_include_directories(nalu SYSTEM PUBLIC ${YAML_CPP_INCLUDE_DIR})
else()
  # YAML 0.5.3 is used
  find_library(YAML_CPP_LIBRARIES NAMES yaml-cpp PATHS ${YAML_DIR}/lib)
  find_path(YAML_CPP_INCLUDE_DIR yaml.h PATHS ${YAML_DIR}/include/yaml-cpp)
  if((DEFINED YAML_CPP_LIBRARIES) AND (DEFINED YAML_CPP_INCLUDE_DIR))
    target_include_directories(nalu SYSTEM PUBLIC ${YAML_CPP_INCLUDE_DIR}/..)
    set(YAML-CPP_FOUND TRUE)
  endif()
endif()
if(YAML-CPP_FOUND)
  message(STATUS "Found YAML-CPP = ${YAML_DIR}")
  target_link_libraries(nalu PUBLIC ${YAML_CPP_LIBRARIES})
else()
  message(FATAL_ERROR "YAML-CPP NOT FOUND")
endif()

########################## OpenMP ####################################
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(nalu PUBLIC $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
  target_link_libraries(nalu PUBLIC $<$<BOOL:${OpenMP_Fortran_FOUND}>:OpenMP::OpenMP_Fortran>)
endif()

############################ CUDA ######################################
if(ENABLE_CUDA)
  target_compile_definitions(nalu PUBLIC USE_STK_SIMD_NONE)
  separate_arguments(Trilinos_CXX_COMPILER_FLAGS)
  target_compile_options(nalu PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${Trilinos_CXX_COMPILER_FLAGS}>)
  target_compile_options(nalu PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
endif()

############################ FFTW ######################################
if(ENABLE_FFTW)
  set(CMAKE_PREFIX_PATH ${FFTW_DIR} ${CMAKE_PREFIX_PATH})
  find_package(FFTW QUIET REQUIRED)
  if(FFTW_FOUND)
    message(STATUS "Found FFTW = ${FFTW3_LIBRARY_DIRS}")
    target_link_libraries(nalu PUBLIC ${FFTW_LIBRARIES})
    target_include_directories(nalu SYSTEM PUBLIC ${FFTW_INCLUDE_DIRS})
    target_compile_definitions(nalu PUBLIC NALU_USES_FFTW)
  endif()
endif()

############################ HYPRE #####################################
if(ENABLE_HYPRE)
  set(CMAKE_PREFIX_PATH ${HYPRE_DIR} ${CMAKE_PREFIX_PATH})
  find_package(HYPRE QUIET REQUIRED)
  if(HYPRE_FOUND)
    message(STATUS "Found HYPRE = ${HYPRE_DIR}")
    target_link_libraries(nalu PUBLIC ${HYPRE_LIBRARIES})
    target_include_directories(nalu SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS})
    target_compile_definitions(nalu PUBLIC NALU_USES_HYPRE)
    include(CheckCXXSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES "${HYPRE_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_LIBRARIES "${HYPRE_LIBRARIES}")
    check_cxx_symbol_exists(
      HYPRE_BIGINT "${HYPRE_INCLUDE_DIRS}/HYPRE_config.h" NALU_HYPRE_BIGINT)
    check_cxx_symbol_exists(
      hypre_KRYLOV_COGMRES_HEADER "${HYPRE_INCLUDE_DIRS}/krylov.h" NALU_HYPRE_COGMRES)
    if(NOT NALU_HYPRE_BIGINT)
      message(WARNING
        "HYPRE does not enable 64-bit integer support; will fail on large problems!")
    endif()
    if (NALU_HYPRE_COGMRES)
      target_compile_definitions(nalu PUBLIC HYPRE_COGMRES)
    else()
      message(STATUS "Disabling HYPRE CO-GMRES interface")
    endif()
  endif()
endif()

########################## OPENFAST ####################################
if(ENABLE_OPENFAST)
  set(CMAKE_PREFIX_PATH ${OpenFAST_DIR} ${CMAKE_PREFIX_PATH})
  find_package(OpenFAST QUIET REQUIRED)
  if(OpenFAST_FOUND)
    message(STATUS "Found OpenFAST = ${OpenFAST_LIBRARY_DIRS}")
    target_link_libraries(nalu PUBLIC ${OpenFAST_LIBRARIES} ${OpenFAST_CPP_LIBRARIES})
    target_include_directories(nalu SYSTEM PUBLIC ${OpenFAST_INCLUDE_DIRS})
    target_compile_definitions(nalu PUBLIC NALU_USES_OPENFAST)
  endif()
endif()

############################ TIOGA #####################################
if(ENABLE_TIOGA)
  set(CMAKE_PREFIX_PATH ${TIOGA_DIR} ${CMAKE_PREFIX_PATH})
  find_package(TIOGA QUIET REQUIRED)
  if(TIOGA_FOUND)
    message(STATUS "Found TIOGA = ${TIOGA_LIBRARY_DIRS}")
    target_link_libraries(nalu PUBLIC ${TIOGA_LIBRARIES})
    target_include_directories(nalu SYSTEM PUBLIC ${TIOGA_INCLUDE_DIRS})
    target_compile_definitions(nalu PUBLIC NALU_USES_TIOGA)
    if(TIOGA_HAS_NODEGID)
      target_compile_definitions(nalu PUBLIC TIOGA_HAS_NODEGID)
    endif()
  endif()
endif()

########################### NALU #####################################
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Add -restrict to Intel CXX compiler
target_compile_options(nalu PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Intel>>:-restrict>)
# Or with CMake >= 3.15
#target_compile_options(nalu PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,Intel>:-restrict>)

# Logic for handling warnings
if(ENABLE_ALL_WARNINGS)
  # GCC, Clang, and Intel seem to accept these
  list(APPEND NALU_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # ifort doesn't like -Wall
    list(APPEND NALU_Fortran_FLAGS "-Wall")
  else()
    # Intel always reports some diagnostics we don't necessarily care about
    list(APPEND NALU_CXX_FLAGS "-diag-disable:11074,11076")
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
    # Avoid notes about -faligned-new with GCC > 7
    list(APPEND NALU_CXX_FLAGS "-faligned-new")
  endif()
endif()

# Add our extra flags according to language
separate_arguments(NALU_CXX_FLAGS)
separate_arguments(NALU_Fortran_FLAGS)
target_compile_options(nalu PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NALU_CXX_FLAGS}>)
target_compile_options(nalu PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${NALU_Fortran_FLAGS}>)
target_compile_options(nalu PUBLIC $<$<BOOL:${ENABLE_WERROR}>:-Werror>)

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

target_include_directories(nalu PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(nalu PUBLIC ${CMAKE_BINARY_DIR}/include)

# Most linking, etc, is set to PUBLIC for libnalu, so we merely link to libnalu for the exes
target_link_libraries(${nalu_ex_name} PRIVATE nalu)
target_link_libraries(${utest_ex_name} PRIVATE nalu)
target_include_directories(${utest_ex_name} PRIVATE "${CMAKE_SOURCE_DIR}/unit_tests")

add_subdirectory(src)
add_subdirectory(unit_tests)

set(nalu_ex_catalyst_name "naluXCatalyst")
if(ENABLE_PARAVIEW_CATALYST)
   set(PARAVIEW_CATALYST_INSTALL_PATH "" CACHE PATH
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

if(ENABLE_EXAMPLES)
   add_subdirectory(examples)
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
