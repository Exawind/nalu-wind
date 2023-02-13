target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestWienerMilenkovic.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestDisplacements.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestDeflectionRamping.C
)

if(ENABLE_OPENFAST_FSI)
target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestFSIpt2line.C
)
endif()

