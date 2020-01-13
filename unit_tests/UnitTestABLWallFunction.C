#include <gtest/gtest.h>
#include <memory>

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"
#include "UnitTestUtils.h"

#include "ABLProfileFunction.h"
#include "AssembleElemSolverAlgorithm.h"
#include "AssembleMomentumElemABLWallFunctionSolverAlgorithm.h"
#include "EquationSystem.h"
#include "master_element/MasterElement.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu{

struct HelperObjectsABLWallFunction {
  HelperObjectsABLWallFunction(stk::mesh::BulkData& bulk, int numDof, stk::mesh::Part* part, const double &z0, const double &Tref, const double &gravity, stk::topology topo)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    eqSystems(realm),
    eqSystem(eqSystems),
    linsys(new unit_test_utils::TestLinearSystem(realm, numDof, &eqSystem, topo)),
    z0_(z0),
    Tref_(Tref),
    gravity_(gravity)
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    eqSystem.linsys_ = linsys;
    elemABLWallFunctionSolverAlg.reset(
      new AssembleMomentumElemABLWallFunctionSolverAlgorithm(
        realm, part, &eqSystem, false, gravity_, z0_, Tref_));
    geomBndryAlg.reset(new GeometryBoundaryAlg<AlgTraitsQuad4>(realm, part));
  }

  ~HelperObjectsABLWallFunction()
  {
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  std::unique_ptr<unit_test_utils::NaluTest> naluObj;
  sierra::nalu::Realm& realm;
  sierra::nalu::EquationSystems eqSystems;
  sierra::nalu::EquationSystem eqSystem;
  unit_test_utils::TestLinearSystem* linsys;
  std::unique_ptr<AssembleMomentumElemABLWallFunctionSolverAlgorithm> elemABLWallFunctionSolverAlg;
  std::unique_ptr<GeometryBoundaryAlg<AlgTraitsQuad4>> geomBndryAlg;
  const double z0_;
  const double Tref_;
  const double gravity_;
};

#ifndef KOKKOS_ENABLE_CUDA

/* This test creates and calls the ABL wall function element algorithm
   for a single-element hex8 mesh and evaluates the resulting rhs vector
   for one of the faces against a pre-calculated value.
*/
TEST_F(ABLWallFunctionHex8ElementWithBCFields, abl_wall_function_elem_alg_rhs) {
  const double z0 = 0.1;
  const double Tref = 300.0;
  const double gravity = 9.81;
  const double rho_specified = 1.0;
  const double utau_specified = 0.067118435077841;
  const double up_specified = 0.15;
  const double yp_specified = 0.25;
  const double aMag = 0.25;
  const double tolerance = 1.0e-12;
  const int numDof = 3;

  SetUp(rho_specified, utau_specified, up_specified, yp_specified);
  double rhs_gold = -rho_specified*utau_specified*utau_specified*aMag;
  HelperObjectsABLWallFunction helperObjs(bulk, numDof, &meta.universal_part(), z0, Tref, gravity, stk::topology::HEX_8);
  helperObjs.geomBndryAlg->execute();

  // Element alg test
  helperObjs.elemABLWallFunctionSolverAlg->initialize_connectivity();
  helperObjs.elemABLWallFunctionSolverAlg->execute();

  unit_test_utils::TestLinearSystem *linsys = helperObjs.linsys;

  EXPECT_NEAR(linsys->rhs_(0), rhs_gold, tolerance);
}

#endif

}
}
