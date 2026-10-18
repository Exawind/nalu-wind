#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================

macro(setup_test testname np)
  set(TEST_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}")
  file(MAKE_DIRECTORY ${TEST_WORKING_DIR})
  if(KYNEMA_UGF_SAVE_GOLDS)
    file(MAKE_DIRECTORY ${SAVED_GOLDS_DIR}/${testname})
    set(SAVE_GOLDS_COMMAND "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}.norm.gold")
    set(SAVE_GOLDS_COMMAND_RST "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}_rst.norm.gold")
    set(SAVE_GOLDS_COMMAND_NP "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}Np${np}.norm.gold")
    set(SAVE_GOLDS_COMMAND_NC "${SAVED_GOLDS_DIR}/${testname}/${testname}.nc.gold")
  endif()
  set(MPI_PREAMBLE "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}")
  set(MPI_COMMAND "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${kynema_ugf_ex_name} ${MPIEXEC_POSTFLAGS}")
  set(INPUT_FILE_BASE "${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}")
  set(INPUT_FILE "${INPUT_FILE_BASE}.yaml")
  set(INPUT_FILE_RST "${INPUT_FILE_BASE}_rst.yaml")
  set(INPUT_FILE_R0 "${INPUT_FILE_BASE}_R0.yaml")
  set(INPUT_FILE_R1 "${INPUT_FILE_BASE}_R1.yaml")
  set(OUTPUT_FILE "${testname}.log")
  set(OUTPUT_FILE_RST "${testname}_rst.log")
  set(OUTPUT_FILE_R0 "${testname}_R0.log")
  set(OUTPUT_FILE_R1 "${testname}_R1.log")
  set(OUTPUT_FILE_NP "${testname}Np${np}.log")
  set(RUN_COMMAND "${MPI_COMMAND} -i ${INPUT_FILE} -o ${OUTPUT_FILE} 2>&1 && cat ${OUTPUT_FILE}")
  set(RUN_COMMAND_RST "${MPI_COMMAND} -i ${INPUT_FILE_RST} -o ${OUTPUT_FILE_RST}")
  set(RUN_COMMAND_R0 "${MPI_COMMAND} -i ${INPUT_FILE_R0} -o ${OUTPUT_FILE_R0}")
  set(RUN_COMMAND_R1 " && ${MPI_COMMAND} -i ${INPUT_FILE_R1} -o ${OUTPUT_FILE_R1}")
  set(RUN_COMMAND_NP "${MPI_COMMAND} -i ${INPUT_FILE} -o ${OUTPUT_FILE_NP}")
  set(COMPARE_GOLDS_COMMAND_BASE " && ${CMAKE_CURRENT_SOURCE_DIR}/check_norms.py --abs-tol ${TEST_ABS_TOL} --rel-tol ${TEST_REL_TOL}")
  set(COMPARE_GOLDS_COMMAND "${COMPARE_GOLDS_COMMAND_BASE} ${testname} ${KYNEMA_UGF_REFERENCE_GOLDS_DIR}/${testname}/${testname}.norm.gold ${SAVE_GOLDS_COMMAND}")
  set(COMPARE_GOLDS_COMMAND_RST "${COMPARE_GOLDS_COMMAND_BASE} ${testname}_rst ${KYNEMA_UGF_REFERENCE_GOLDS_DIR}/${testname}/${testname}_rst.norm.gold ${SAVE_GOLDS_COMMAND_RST}")
  set(COMPARE_GOLDS_COMMAND_NP "${COMPARE_GOLDS_COMMAND_BASE} ${testname}Np${np} ${KYNEMA_UGF_REFERENCE_GOLDS_DIR}/${testname}/${testname}Np${np}.norm.gold ${SAVE_GOLDS_COMMAND_NP}")
  set(COMPARE_GOLDS_COMMAND_NC " && ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/passfail.sh ${testname}.nc ${KYNEMA_UGF_REFERENCE_GOLDS_DIR}/${testname}/${testname}.nc.gold ${SAVE_GOLDS_COMMAND_NC}")
  set(CHECK_SOL_NORMS_COMMAND " && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/check_sol_norms.py ${testname} ${KYNEMA_UGF_REFERENCE_GOLDS_DIR}/${testname}/${testname}.norm.gold --abs-tol ${TEST_ABS_TOL} ${SAVE_GOLDS_COMMAND}")
  set(COMPARE_GOLDS_COMMAND_ERRORS " && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/norms.py")
endmacro(setup_test)

macro(set_properties testname cost)
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
      set(TEST_TIMEOUT 28800)
    else()
      set(TEST_TIMEOUT 1800)
    endif()
    set_tests_properties(${testname} PROPERTIES TIMEOUT ${TEST_TIMEOUT} PROCESSORS ${np} WORKING_DIRECTORY "${TEST_WORKING_DIR}")
    if(NOT "${cost}" STREQUAL "")
      set_tests_properties(${testname} PROPERTIES COST ${cost})
    endif()
endmacro(set_properties)

# Standard regression test
function(add_test_r testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND}")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "regression" ATTACHED_FILES "${OUTPUT_FILE}")
endfunction(add_test_r)

# Regression test with single restart
function(add_test_r_rst testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND}; ${RUN_COMMAND_RST}${COMPARE_GOLDS_COMMAND_RST}")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "regression" ATTACHED_FILES "${OUTPUT_FILE_RST}")
endfunction(add_test_r_rst)

# Regression test with postprocessing
function(add_test_r_post testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND_NC}")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "regression" ATTACHED_FILES "${OUTPUT_FILE}")
endfunction(add_test_r_post)

# Verification test comparing solution norms
function(add_test_v_sol_norm testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${CHECK_SOL_NORMS_COMMAND}")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "verification" ATTACHED_FILES "${OUTPUT_FILE}")
endfunction(add_test_v_sol_norm)

# Verification test with two resolutions
function(add_test_v2 testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND_R0}${RUN_COMMAND_R1}${COMPARE_GOLDS_COMMAND_ERRORS}")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "verification" ATTACHED_FILES "${OUTPUT_FILE_R0};${OUTPUT_FILE_R1}")
endfunction(add_test_v2)

# Regression test that runs with different numbers of processes
function(add_test_r_np testname np)
    setup_test(${testname} ${np})
    add_test(${testname}Np${np} sh -c "${RUN_COMMAND_NP}${COMPARE_GOLDS_COMMAND_NP}")
    set_properties(${testname}Np${np} "${ARGV2}")
    set_tests_properties(${testname}Np${np} PROPERTIES LABELS "regression" ATTACHED_FILES "${OUTPUT_FILE_NP}")
endfunction(add_test_r_np)

# Standard unit test
function(add_test_u testname np)
    setup_test(${testname} ${np})
    if(${np} EQUAL 1)
      set(GTEST_SHUFFLE "--gtest_shuffle")
    else()
      unset(GTEST_SHUFFLE)
    endif()
    add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${MPIEXEC_POSTFLAGS} ${GTEST_SHUFFLE} 2>&1")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "unit")
    if(ENABLE_OPENFAST)
      # create symlink to nrelmw.fst
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
        ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/nrel5mw.fst
      )
    endif()
endfunction(add_test_u)

# GPU unit test
function(add_test_u_gpu testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${MPIEXEC_POSTFLAGS} 2>&1")
    set_properties(${testname} "${ARGV2}")
    set_tests_properties(${testname} PROPERTIES LABELS "unit")
    if(ENABLE_OPENFAST)
      # create symlink to nrelmw.fst
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
        ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/nrel5mw.fst
      )
    endif()
endfunction(add_test_u_gpu)

# Regression test with catalyst capability
#function(add_test_r_cat testname np ncat)
#    if(ENABLE_PARAVIEW_CATALYST)
#      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.template.yaml)
#        setup_test(${testname} ${np})
#        add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${kynema_ugf_ex_catalyst_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/${testname}_catalyst.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail_catalyst.sh ${testname} ${ncat}")
#        set_properties(${testname})
#        set_tests_properties(${testname} PROPERTIES LABELS "regression")
#        set(CATALYST_FILE_INPUT_DECK_COMMAND "catalyst_file_name: catalyst.txt")
#        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.template.yaml
#                       ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/${testname}_catalyst.yaml @ONLY)
#        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/catalyst.txt
#             DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
#      endif()
#    else()
#      add_test_r(${testname} ${np})
#    endif()
#endfunction(add_test_r_cat)

if(NOT ENABLE_CUDA AND NOT ENABLE_ROCM)

  #=============================================================================
  # Regression tests
  #=============================================================================

  if (ENABLE_TRILINOS_SOLVERS)
    add_test_r(ActLineSimpleFLLC 4 3056.66)
    add_test_r(ActLineSimpleNGP 2 7043.47)
    add_test_r(airfoilSSTSUST 4 7197.49)
    add_test_r(ablUnstableEdge 4 8699.51)
    add_test_r(ablUnstableEdge_ra 4 7062.9)
    add_test_r(ablStableEdge 4 6275.73)
    add_test_r_post(ablNeutralStat 8 4346.72)
    add_test_r(ablNeutralEdge 8 8551.7)
    add_test_r(ablNeutralEdgeSegregated 8 8175.64)
    add_test_r(ablNeutralEdgeNoSlip 4 8222.25)
    add_test_r(ablHill3dSymPenalty 4 9268.67)
    add_test_r(ablNeutralNGPTrilinos 2 22364.8)
    add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1 94.54)
    #add_test_r(conduction_p4 4)
    add_test_r(dgNonConformalEdgeCylinder 8 8414.8)
    add_test_r(dgNonConformalFluidsEdge 4 419.34)
    add_test_r(drivenCavity_p1 4 1115.4)
    add_test_r(edgeHybridFluids 8 9752.66)
    add_test_r(ekmanSpiral 4 379.13)
    add_test_r_rst(heatedWaterChannelEdge 4 1068.3)
    add_test_r(karmanVortex 1 351.82)
    add_test_r(nonIsoEdgeOpenJet 4 7239.82)
    add_test_r(MeshMotionInterior 4 260.78)
    add_test_r_np(periodic3dEdge 1 2315.86)
    add_test_r_np(periodic3dEdge 4 953.86)
    add_test_r_np(periodic3dEdge 8 646.09)
    add_test_r(taylorGreenVortex_p3 4 10812.3)
    add_test_r(vortexOpen 4 1762.22)
    add_test_r(VOFDroplet 4 14741.8)
    add_test_r(VOFInertialDroplet 4 12971.4)

    if (ENABLE_FFTW)
      add_test_r(ablHill3d_pp 4)
      add_test_r(ablHill3d_ip 4)
      add_test_r(ablHill3d_ii 4)
    endif()
  endif()

  if (ENABLE_MATRIXFREE)
    add_test_r(drivenCavity_p1 4 1115.4)
    #add_test_r(conduction_p4 4)
    add_test_r(taylorGreenVortex_p3 4 10812.3)
  endif()

  if(ENABLE_HYPRE)
    add_test_r(ablNeutralNGPHypre 2 2205.42)
    add_test_r(ablNeutralNGPHypreSegregated 2 1178.16)
    add_test_r(airfoilRANSEdgeNGPHypre.rst 2 39.22)
    add_test_r(fsiTurbineSurrogate 4 312.38)
    add_test_r(VOFZalDisk 4 259.47)
    add_test_r(airfoilSST_Gamma_Trans 4 117.18)
    add_test_r(airfoilSA 4 117.18)
    add_test_r(airfoilSA_AFT 4 117.18)
    add_test_r(IDDESPeriodicHillEdge 8 6300.87)
    add_test_r(IDDESTransPeriodicHillEdge 8 6215.53)

    if (ENABLE_TRILINOS_SOLVERS)
      add_test_r(aslNeutralEdgeSST 4 199.32)
      add_test_r_rst(amsChannelEdge 4)
      add_test_r(ablNeutralEdgeSST 4 521.64)
      add_test_r(ablNeutralEdgeAMS 4 478.11)
      add_test_r(KOChannelEdge 4 451.04)
      add_test_r(KEChannelEdge 4 454.73)
      add_test_r_rst(SSTAMSChannelEdge 4 1380.93)
      #add_test_r_rst(SSTAMSOversetRotCylinder 2)
      add_test_r(SSTChannelEdge 4 437.89)
      add_test_r(SSTLRChannelEdge 4 460.74)
      add_test_r(SSTPeriodicHillEdge 4 13877.4)
      add_test_r(SSTWallHumpEdge 4 9654.44)
    endif()

    add_test_r(multiElemCylinder 4 124.4)
  endif()

  if(ENABLE_OPENFAST AND ENABLE_HYPRE)
     add_test_r(nrel5MWactuatorLine 4 1456.46)
     add_test_r(nrel5MWactuatorLineAnisoGauss 4 1593.83)
     add_test_r(nrel5MWactuatorLineFllc 4 1601.2)
     add_test_r(nrel5MWactuatorDisk 4 1039.87)
     add_test_r(nrel5MWadvActLine 4 1438.14)
     add_subdirectory(test_files/nrel5MWactuatorLine)
  endif()

  if(ENABLE_TIOGA AND ENABLE_TRILINOS_SOLVERS)
    add_test_r(oversetSphereTIOGA 8 5433.3)
    add_test_r(oversetRotCylinder 4 4200.39)
    add_test_r(oversetCylNGPTrilinos 2 5567.31)
    add_test_r(oversetRotCylNGPTrilinos 2 10012.4)
  endif()

  if (ENABLE_TIOGA AND ENABLE_HYPRE)
    add_test_r(oversetRotCylNGPHypre 2 692.09)
    add_test_r(oversetOscCylNGPHypre 2 687.22)
    add_test_r(oversetPrescribedCylinder 4 1246.11)
    if (ENABLE_TRILINOS_SOLVERS)
      add_test_r(oversetRotCylinderHypre 2 4609.43)
      add_test_r(oversetRotCylMultiRealm 2 3575.27)
      add_test_r_rst(oversetMovingCylinder 4 9907.7)
      add_test_r(oversetTransformedTwoRotCylinder 1 572.73)
    endif()
  endif()

  #=============================================================================
  # Comparing solution norm tests
  #=============================================================================
  if(ENABLE_HYPRE)
    add_test_v_sol_norm(convTaylorVortex 2 139.2)
  endif()

  #=============================================================================
  # Convergence tests
  #=============================================================================
  if(ENABLE_TRILINOS_SOLVERS)
    add_test_v2(BoussinesqNonIso 8 15135.8)
  endif()

  #=============================================================================
  # Unit tests
  #=============================================================================
  add_test_u(unitTest1 1 6317.62)
  #add_test_u(unitTest2 2) # Can hang indefinitely

  #=============================================================================
  # Performance tests
  #=============================================================================

else()

  #=============================================================================
  # Regression tests
  #=============================================================================

  if (ENABLE_TRILINOS_SOLVERS)
    add_test_r(ActLineSimpleNGP 2 7043.47)
    add_test_r(ablNeutralNGPTrilinos 2 22364.8)
    add_test_r(ActLineSimpleFLLC 2 3056.66)
    add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1 94.54)
  endif()

  if(ENABLE_OPENFAST)
    add_test_r(nrel5MWactuatorLine 2 1456.46)
    add_subdirectory(test_files/nrel5MWactuatorLine)
  endif()

  if(ENABLE_HYPRE)
    add_test_r(fsiTurbineSurrogate 2 312.38)
    add_test_r(airfoilRANSEdgeNGPHypre.rst 2 39.22)
    add_test_r(ablNeutralNGPHypre 2 2205.42)
    add_test_r(ablNeutralNGPHypreSegregated 2 1178.16)
    add_test_r(multiElemCylinder 2 124.4)
    add_test_r(VOFZalDisk 2 259.47)
    add_test_r(airfoilSST_Gamma_Trans 2 117.18)
    add_test_r(airfoilSA 2 117.18)
    add_test_r(airfoilSA_AFT 2 117.18)
    add_test_r(IDDESPeriodicHillEdge 2 6300.87)
    add_test_r(IDDESTransPeriodicHillEdge 2 6215.53)
  endif()

  if (ENABLE_TIOGA AND ENABLE_TRILINOS_SOLVERS)
    add_test_r(oversetCylNGPTrilinos 2 5567.31)
    add_test_r(oversetRotCylNGPTrilinos 2 10012.4)
  endif()

  if (ENABLE_TIOGA AND ENABLE_HYPRE)
    add_test_r(oversetRotCylNGPHypre 2 692.09)
  endif()

  if(ENABLE_MATRIXFREE)
    #add_test_r(conduction_p4 2)
    add_test_r(taylorGreenVortex_p3 2 10812.3)
    add_test_r(drivenCavity_p1 2 1115.4)
  endif()

  #=============================================================================
  # Comparing solution norm tests
  #=============================================================================
  if(ENABLE_HYPRE)
    add_test_v_sol_norm(convTaylorVortex 2 139.2)
  endif()

  #=============================================================================
  # GPU unit tests
  #=============================================================================
  add_test_u_gpu(unitTestGPU 1)

endif()
