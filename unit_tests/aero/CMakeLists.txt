target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestWienerMilenkovic.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestDisplacements.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestDeflectionRamping.C
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestPt2Line.C
)

if(ENABLE_OPENFAST)
target_sources(${utest_ex_name} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestFSIturbine.C
)
endif()

