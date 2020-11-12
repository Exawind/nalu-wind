#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================

# Standard regression test
function(add_test_r testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname} ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}.norm.gold")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_r)

# Standard performance test
function(add_test_p testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname} ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}.norm.gold")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "performance")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_p)

# Regression test with single restart
function(add_test_r_rst testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname} ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}.norm.gold; ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_rst.yaml -o ${testname}_rst.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname}_rst ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}_rst.norm.gold")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_r_rst)

# Regression test with postprocessing 
function(add_test_r_post testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/passfail ")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 10800 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_r_post)

# Regression test with input
function(add_test_r_inp testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname} ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}.norm.gold; ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_Input.yaml -o ${testname}_Input.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname}_Input ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}_Input.norm.gold")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_r_inp)

# Verification test with three resolutions
function(add_test_v3 testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_R0.yaml -o ${testname}_R0.log && ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_R1.yaml -o ${testname}_R1.log && ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_R2.yaml -o ${testname}_R2.log && python ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/norms.py")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "verification")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_v3)

# Verification test with two resolutions
function(add_test_v2 testname np)
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_R0.yaml -o ${testname}_R0.log && ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}_R1.yaml -o ${testname}_R1.log && python ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/norms.py")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "verification")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_v2)

# Regression test that runs with different numbers of processes
function(add_test_r_np testname np)
    add_test(${testname}Np${np} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_name} ${MPIEXEC_POSTFLAGS} -i ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.yaml -o ${testname}Np${np}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail.py --abs-tol ${TOLERANCE} ${testname}Np${np} ${NALU_GOLD_NORMS_DIR}/test_files/${testname}/${testname}Np${np}.norm.gold")
    set_tests_properties(${testname}Np${np} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
endfunction(add_test_r_np)

# Standard unit test
function(add_test_u testname np)
    if(${np} EQUAL 1)
      set(GTEST_SHUFFLE "--gtest_shuffle")
    else()
      unset(GTEST_SHUFFLE)
    endif()
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${GTEST_SHUFFLE}")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 6000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "unit")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
    # create symlink to nrelmw.fst 
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
      ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/nrel5mw.fst
    )
endfunction(add_test_u)

# GPU unit test
function(add_test_u_gpu testname np)
    set(FILTER "--gtest_filter=BasicKokkos.discover_execution_space:*.NGP*")
    add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${utest_ex_name} ${FILTER}")
    set_tests_properties(${testname} PROPERTIES TIMEOUT 6000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "unit")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
    # create symlink to nrelmw.fst 
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_BINARY_DIR}/reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst
      ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/nrel5mw.fst
    )
endfunction(add_test_u_gpu)

# Regression test with catalyst capability
function(add_test_r_cat testname np ncat)
    if(ENABLE_PARAVIEW_CATALYST)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${testname}/${testname}.template.yaml)
        add_test(${testname} sh -c "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} ${CMAKE_BINARY_DIR}/${nalu_ex_catalyst_name} -i ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}/${testname}_catalyst.yaml -o ${testname}.log && ${CMAKE_CURRENT_SOURCE_DIR}/pass_fail_catalyst.sh ${testname} ${ncat}")
        set_tests_properties(${testname} PROPERTIES TIMEOUT 18000 PROCESSORS ${np} WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname}" LABELS "regression")
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_files/${testname})
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
  add_test_r(ablStableElem 4)
  add_test_r(ablUnstableEdge 4)
  add_test_r(ablUnstableEdge_ra 4)
  add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1)
  add_test_r(conduction_p4 4)
  add_test_r(cvfemHC 8)
  add_test_r(dgMMS 6)
  add_test_r(dgNonConformal 4)
  add_test_r(dgNonConformal3dFluids 4)
  add_test_r(dgNonConformal3dFluidsHexTet 4)
  add_test_r(dgNonConformal3dFluidsP1P2 8)
  add_test_r(dgNonConformalEdge 4)
  add_test_r(dgNonConformalEdgeCylinder 8)
  add_test_r(dgNonConformalElemCylinder 8)
  add_test_r(dgNonConformalFluids 4)
  add_test_r(dgNonConformalFluidsEdge 4)
  add_test_r_rst(dgNonConformalThreeBlade 4)
  add_test_r(drivenCavity_p1 4)
  add_test_r(ductElemWedge 2)
  add_test_r(ductWedge 2)
  add_test_r(edgeHybridFluids 8)
  add_test_r(ekmanSpiral 4)
  add_test_r(ekmanSpiralConsolidated 4)
  add_test_r_inp(elemBackStepLRSST 4)
  add_test_r(elemClosedDomain 2)
  add_test_r(elemHybridFluids 8)
  add_test_r(elemHybridFluidsShift 8)
  add_test_r(femHC 2)
  add_test_r(femHCGL 2)
  add_test_r(heatedBackStep 4)
  add_test_r_rst(heatedWaterChannelEdge 4)
  add_test_r(heatedWaterChannelElem 4)
  add_test_r_rst(heliumPlume 8)
  add_test_r(hoHelium 8)
  add_test_r(hoVortex 2)
  add_test_r(karmanVortex 1)
  add_test_r(milestoneRun 4)
  add_test_r(milestoneRunConsolidated 4)
  add_test_r_cat(mixedTetPipe 8 7)
  add_test_r(movingCylinder 4)
  add_test_r(nonConformalWithPeriodic 2)
  add_test_r(nonConformalWithPeriodicConsolidated 2)
  add_test_r(nonIsoEdgeOpenJet 4)
  add_test_r(nonIsoElemOpenJet 4)
  add_test_r(nonIsoElemOpenJetConsolidated 4)
  add_test_r(nonIsoNonUniformEdgeOpenJet 4)
  add_test_r(nonIsoNonUniformElemOpenJet 4)
  add_test_r_np(periodic3dElem 1)
  add_test_r_np(periodic3dElem 4)
  add_test_r_np(periodic3dElem 8)
  add_test_r_np(periodic3dEdge 1)
  add_test_r_np(periodic3dEdge 4)
  add_test_r_np(periodic3dEdge 8)
  add_test_r(quad9HC 2)
  add_test_r_cat(steadyTaylorVortex 4 6)
  add_test_r(taylorGreenVortex_p3 4)
  add_test_r(variableDensNonIso 2)
  add_test_r(variableDensNonUniform 2)
  add_test_r(vortexOpen 4)
  add_test_r(ActLineSimple 4)
  add_test_r(ActLineSimpleFLLC 4)
  add_test_r(ActLineSimpleNGP 2)

  if (ENABLE_FFTW)
    add_test_r(ablHill3d_pp 4)
    add_test_r(ablHill3d_ip 4)
    add_test_r(ablHill3d_ii 4)
  endif()

  if(ENABLE_HYPRE)
    add_test_r(airfoilRANSEdgeNGP 2)
    #add_test_r(airfoilRANSElem 2)
    #add_test_r(dgncThreeBladeHypre 2)
    add_test_r_rst(tamsChannelEdge 4)
    add_test_r(SSTChannelEdge 4)
    add_test_r_rst(SSTTAMSChannelEdge 4)
    add_test_r_rst(SSTTAMSOversetRotCylinder 2)
    add_test_r(ablNeutralNGPHypre 2)
    add_test_r(ablNeutralNGPHypreSegregated 2)
    add_test_r(airfoilRANSEdgeNGPHypre 2)
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
  endif()

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
  add_test_p(uqSlidingMeshDG 8)
  add_test_p(waleElemXflowMixFrac3.5m 8)

else(NOT ENABLE_CUDA)

  #=============================================================================
  # Regression tests
  #=============================================================================
  add_test_r(ablNeutralNGPTrilinos 2)
  add_test_r(conduction_p4 2)
  add_test_r(airfoilRANSEdgeNGPTrilinos.rst 1)
  add_test_r(ActLineSimpleNGP 2)
  add_test_r(taylorGreenVortex_p3 2)
  add_test_r(drivenCavity_p1 2)

  if(ENABLE_OPENFAST)
    add_test_r(nrel5MWactuatorLine 2)
    add_subdirectory(test_files/nrel5MWactuatorLine)
  endif()

  if(ENABLE_HYPRE)
    add_test_r(airfoilRANSEdgeNGP 2)
    add_test_r(airfoilRANSEdgeNGPHypre 2)
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
  # GPU unit tests
  #=============================================================================
  add_test_u_gpu(unitTestGPU 1)

endif(NOT ENABLE_CUDA)
