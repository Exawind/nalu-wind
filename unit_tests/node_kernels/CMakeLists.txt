target_sources(${utest_ex_name} PRIVATE
  UnitTestContinuityGclNodeKernel.C
  UnitTestContinuityMassBDFNodeKernel.C
  UnitTestEnthalpyABLForceNode.C
  UnitTestMomentumABLForceNode.C
  UnitTestMomentumBodyForceNode.C
  UnitTestMomentumBoussinesqNode.C
  UnitTestMomentumGclSrcNodeKernel.C
  UnitTestMomentumMassBDFNodeKernel.C
  UnitTestMomentumCoriolisNode.C
  UnitTestScalarGclNodeKernel.C
  UnitTestScalarMassBDFNodeKernel.C
  UnitTestMomentumActuatorNodeKernel.C
  UnitTestTurbKineticEnergyRodiNodeKernel.C
  UnitTestSSTNodeKernel.C
  UnitTestKsgsNodeKernel.C
  UnitTestWallDistNode.C
)
