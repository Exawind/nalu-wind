# Copy directory with necessary OpenFAST files to binary directory before running the test
# since we don't want OpenFAST output to dirty the source directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/5MW_Baseline DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

# Set name for library using the prefix and suffix for the current system
set(NREL5MW_DISCON_LIB "${CMAKE_SHARED_LIBRARY_PREFIX}DISCON${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NRELOffshrBsline5MW_Onshore_ServoDyn.dat.in
               ${CMAKE_CURRENT_BINARY_DIR}/5MW_Baseline/NRELOffshrBsline5MW_Onshore_ServoDyn.dat @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nrel5mw.fst.in
               ${CMAKE_CURRENT_BINARY_DIR}/nrel5mw.fst @ONLY)

# Add subdirectory that will actually build the DISCON library
add_subdirectory(5MW_Baseline/ServoData)
