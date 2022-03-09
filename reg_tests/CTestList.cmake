#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================

macro(setup_test testname np)
  set(TEST_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}")
  file(MAKE_DIRECTORY ${TEST_WORKING_DIR})
  if(NALU_WIND_SAVE_GOLDS)
    file(MAKE_DIRECTORY ${SAVED_GOLDS_DIR}/${testname})
    set(SAVE_GOLDS_COMMAND "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}.norm.gold")
    set(SAVE_GOLDS_COMMAND_RST "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}_rst.norm.gold")
    set(SAVE_GOLDS_COMMAND_NP "--save-norm-file ${SAVED_GOLDS_DIR}/${testname}/${testname}Np${np}.norm.gold")
    set(SAVE_GOLDS_COMMAND_NC "${SAVED_GOLDS_DIR}/${testname}/${testname}.nc.gold")
  endif()
  set(MPI_PREAMBLE "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}")
  set(MPI_COMMAND "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS}")
  set(INPUT_FILE_BASE "${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}")
  set(INPUT_FILE "${INPUT_FILE_BASE}.yaml")
  set(INPUT_FILE_RST "${INPUT_FILE_BASE}_rst.yaml")
  set(INPUT_FILE_R0 "${INPUT_FILE_BASE}_R0.yaml")
  set(INPUT_FILE_R1 "${INPUT_FILE_BASE}_R1.yaml")
  set(OUTPUT_FILE "${testname}.log")
  set(OUTPUT_FILE_RST "${testname}_rst.log")
  set(OUTPUT_FILE_R0 "${testname}_R0.log")
  set(OUTPUT_FILE_R1 "${testname}_R0.log")
  set(OUTPUT_FILE_NP "${testname}Np${np}.log")
  set(RUN_COMMAND "${MPI_COMMAND} -i ${INPUT_FILE} -o ${OUTPUT_FILE}")
  set(RUN_COMMAND_RST "${MPI_COMMAND} -i ${INPUT_FILE_RST} -o ${OUTPUT_FILE_RST}")
  set(RUN_COMMAND_R0 "${MPI_COMMAND} -i ${INPUT_FILE_R0} -o ${OUTPUT_FILE_R0}")
  set(RUN_COMMAND_R1 " && ${MPI_COMMAND} -i ${INPUT_FILE_R1} -o ${OUTPUT_FILE_R1}")
  set(RUN_COMMAND_NP "${MPI_COMMAND} -i ${INPUT_FILE} -o ${OUTPUT_FILE_NP}")
  set(COMPARE_GOLDS_COMMAND_BASE " && ${CMAKE_CURRENT_SOURCE_DIR}/check_norms.py --abs-tol ${TEST_ABS_TOL} --rel-tol ${TEST_REL_TOL}")
  set(COMPARE_GOLDS_COMMAND "${COMPARE_GOLDS_COMMAND_BASE} ${testname} ${NALU_WIND_REFERENCE_GOLDS_DIR}/${testname}/${testname}.norm.gold ${SAVE_GOLDS_COMMAND}")
  set(COMPARE_GOLDS_COMMAND_RST "${COMPARE_GOLDS_COMMAND_BASE} ${testname}_rst ${NALU_WIND_REFERENCE_GOLDS_DIR}/${testname}/${testname}_rst.norm.gold ${SAVE_GOLDS_COMMAND_RST}")
  set(COMPARE_GOLDS_COMMAND_NP "${COMPARE_GOLDS_COMMAND_BASE} ${testname}Np${np} ${NALU_WIND_REFERENCE_GOLDS_DIR}/${testname}/${testname}Np${np}.norm.gold ${SAVE_GOLDS_COMMAND_NP}")
  set(COMPARE_GOLDS_COMMAND_NC " && ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/passfail.sh ${testname}.nc ${NALU_WIND_REFERENCE_GOLDS_DIR}/${testname}/${testname}.nc.gold ${SAVE_GOLDS_COMMAND_NC}")
  set(CHECK_SOL_NORMS_COMMAND " && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/check_sol_norms.py ${testname} ${NALU_WIND_REFERENCE_GOLDS_DIR}/${testname}/${testname}.norm.gold --abs-tol ${TEST_ABS_TOL} ${SAVE_GOLDS_COMMAND}")
  set(COMPARE_GOLDS_COMMAND_ERRORS " && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/norms.py")
endmacro(setup_test)

macro(set_properties testname)
    set_tests_properties(${testname} PROPERTIES TIMEOUT 20000 PROCESSORS ${np} WORKING_DIRECTORY "${TEST_WORKING_DIR}")
endmacro(set_properties)

# Standard regression test
function(add_test_r testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "regression")
endfunction(add_test_r)

# Standard performance test
function(add_test_p testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "performance")
endfunction(add_test_p)

# Regression test with single restart
function(add_test_r_rst testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND}; ${RUN_COMMAND_RST}${COMPARE_GOLDS_COMMAND_RST}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "regression")
endfunction(add_test_r_rst)

# Regression test with postprocessing 
function(add_test_r_post testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${COMPARE_GOLDS_COMMAND_NC}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "regression")
endfunction(add_test_r_post)

# Verification test comparing solution norms
function(add_test_v_sol_norm testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND}${CHECK_SOL_NORMS_COMMAND}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "verification")
endfunction(add_test_v_sol_norm)

# Verification test with two resolutions
function(add_test_v2 testname np)
    setup_test(${testname} ${np})
    add_test(${testname} sh -c "${RUN_COMMAND_R0}${RUN_COMMAND_R1}${COMPARE_GOLDS_COMMAND_ERRORS}")
    set_properties(${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "verification")
endfunction(add_test_v2)

# Regression test that runs with different numbers of processes
function(add_test_r_np testname np)
    setup_test(${testname} ${np})
    add_test(${testname}Np${np} sh -c "${RUN_COMMAND_NP}${COMPARE_GOLDS_COMMAND_NP}")
    set_properties(${testname}Np${np})
    set_tests_properties(${testname}Np${np} PROPERTIES LABELS "regression")
endfunction(add_test_r_np)

# Standard unit test
function(add_test_u testname np)
    setup_test(${testname} ${np})
    if(${np} EQUAL 1)
      set(GTEST_SHUFFLE "--gtest_shuffle")
    else()
      unset(GTEST_SHUFFLE)
    endif()
    add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${MPIEXEC_POSTFLAGS} ${GTEST_SHUFFLE}")
    set_properties(${testname})
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
    set(FILTER "--gtest_filter=BasicKokkos.discover_execution_space:*.NGP*")
    add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${MPIEXEC_POSTFLAGS} ${FILTER}")
    set_properties(${testname})
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
function(add_test_r_cat testname np ncat)
    if(ENABLE_PARAVIEW_CATALYST)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.template.yaml)
        setup_test(${testname} ${np})
        add_test(${testname} sh -c "${MPI_PREAMBLE} ${CMAKE_BINARY_DIR}/${nalu_ex_catalyst_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/${testname}_catalyst.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail_catalyst.sh ${testname} ${ncat}")
        set_properties(${testname})
        set_tests_properties(${testname} PROPERTIES LABELS "regression")
        set(CATALYST_FILE_INPUT_DECK_COMMAND "catalyst_file_name: catalyst.txt")
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.template.yaml
                       ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/${testname}_catalyst.yaml @ONLY)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/catalyst.txt
             DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
      endif()
    else()
      add_test_r(${testname} ${np})
    endif()
endfunction(add_test_r_cat)

if(NOT ENABLE_CUDA)

  #=============================================================================
  # Regression tests
  #=============================================================================
  add_test_r_cat(ablNeutralEdge 8 2)
  add_test_r_post(ablNeutralStat 8)
  add_test_r(ablNeutralNGPTrilinos 2)
  add_test_r(ablNeutralEdgeSegregated 8)
  add_test_r(ablStableEdge 4)
  add_test_r(ablNeutralEdgeNoSlip 4)
  add_test_r(ablUnstableEdge 4)
  add_test_r(ablUnstableEdge_ra 4)
  add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1)
  add_test_r(conduction_p4 4)
  add_test_r(dgNonConformalEdgeCylinder 8)
  add_test_r(dgNonConformalFluidsEdge 4)
  add_test_r(drivenCavity_p1 4)
  add_test_r(edgeHybridFluids 8)
  add_test_r(ekmanSpiral 4)
  add_test_r_rst(heatedWaterChannelEdge 4)
  add_test_r(karmanVortex 1)
  add_test_r(nonIsoEdgeOpenJet 4)
  add_test_r_np(periodic3dEdge 1)
  add_test_r_np(periodic3dEdge 4)
  add_test_r_np(periodic3dEdge 8)
  add_test_r(taylorGreenVortex_p3 4)
  add_test_r(vortexOpen 4)
  add_test_r(ActLineSimpleFLLC 4)
  add_test_r(ActLineSimpleNGP 2)
  add_test_r(ablHill3dSymPenalty 4)
  add_test_r(MeshMotionInterior 4)

  if (ENABLE_FFTW)
    add_test_r(ablHill3d_pp 4)
    add_test_r(ablHill3d_ip 4)
    add_test_r(ablHill3d_ii 4)
  endif()

  if(ENABLE_HYPRE)
    add_test_r_rst(amsChannelEdge 4)
    add_test_r(SSTChannelEdge 4)
    add_test_r_rst(SSTAMSChannelEdge 4)
    #add_test_r_rst(SSTAMSOversetRotCylinder 2)
    add_test_r(ablNeutralNGPHypre 2)
    add_test_r(ablNeutralNGPHypreSegregated 2)
    add_test_r(airfoilRANSEdgeNGPHypre.rst 2)
    add_test_r(SSTWallHumpEdge 4)
    add_test_r(SSTPeriodicHillEdge 4)
    add_test_r(IDDESPeriodicHillEdge 4)
  endif(ENABLE_HYPRE)

  if(ENABLE_OPENFAST)
     add_test_r(nrel5MWactuatorLine 4)
     add_test_r(nrel5MWactuatorLineAnisoGauss 4)
     add_test_r(nrel5MWactuatorLineFllc 4)
     add_test_r(nrel5MWactuatorDisk 4)
     add_test_r(nrel5MWadvActLine 4)
     add_subdirectory(test_files/nrel5MWactuatorLine)
  endif(ENABLE_OPENFAST)

  if(ENABLE_TIOGA)
    add_test_r(oversetSphereTIOGA 8)
    add_test_r(oversetRotCylinder 4)
    add_test_r(oversetCylNGPTrilinos 2)
    add_test_r(oversetRotCylNGPTrilinos 2)
  endif(ENABLE_TIOGA)

  if (ENABLE_TIOGA AND ENABLE_HYPRE)
    add_test_r(oversetRotCylNGPHypre 2)
    add_test_r(oversetRotCylinderHypre 2)
    add_test_r(oversetRotCylMultiRealm 2)
    add_test_r_rst(oversetMovingCylinder 4)
  endif()

  #=============================================================================
  # Comparing solution norm tests
  #=============================================================================
  if(ENABLE_HYPRE)
    add_test_v_sol_norm(convTaylorVortex 2)
  endif(ENABLE_HYPRE)

  #=============================================================================
  # Convergence tests
  #=============================================================================
  add_test_v2(BoussinesqNonIso 8)
  
  #=============================================================================
  # Unit tests
  #=============================================================================
  add_test_u(unitTest1 1)
  add_test_u(unitTest2 2)

  #=============================================================================
  # Performance tests
  #=============================================================================

else(NOT ENABLE_CUDA)

  #=============================================================================
  # Regression tests
  #=============================================================================
  add_test_r(ablNeutralNGPTrilinos 2)
  add_test_r(conduction_p4 2)
  add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1)
  add_test_r(ActLineSimpleNGP 2)
  add_test_r(ActLineSimpleFLLC 2)
  add_test_r(taylorGreenVortex_p3 2)
  add_test_r(drivenCavity_p1 2)
  #add_test_r(BLTFlatPlateT3A 4)

  if(ENABLE_OPENFAST)
    add_test_r(nrel5MWactuatorLine 2)
    add_subdirectory(test_files/nrel5MWactuatorLine)
  endif()

  if(ENABLE_HYPRE)
    add_test_r(airfoilRANSEdgeNGPHypre.rst 2)
    add_test_r(ablNeutralNGPHypre 2)
    add_test_r(ablNeutralNGPHypreSegregated 2)
  endif(ENABLE_HYPRE)

  if (ENABLE_TIOGA)
    add_test_r(oversetCylNGPTrilinos 2)
    add_test_r(oversetRotCylNGPTrilinos 2)
  endif()

  if (ENABLE_TIOGA AND ENABLE_HYPRE)
    add_test_r(oversetRotCylNGPHypre 2)
  endif()

  #=============================================================================
  # Comparing solution norm tests
  #=============================================================================
  if(ENABLE_HYPRE)
    add_test_v_sol_norm(convTaylorVortex 2)
  endif(ENABLE_HYPRE)

  #=============================================================================
  # GPU unit tests
  #=============================================================================
  add_test_u_gpu(unitTestGPU 1)

endif(NOT ENABLE_CUDA)
