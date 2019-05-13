#include <gtest/gtest.h>

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"
#include "UnitTestUtils.h"

#include "AssembleElemSolverAlgorithm.h"
#include "ComputeABLWallFrictionVelocityAlgorithm.h"
#include "AssembleMomentumElemABLWallFunctionSolverAlgorithm.h"
#include "AssembleMomentumEdgeABLWallFunctionSolverAlgorithm.h"
#include "ComputeGeometryBoundaryAlgorithm.h"
#include "EquationSystem.h"
#include "master_element/MasterElement.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu{


struct HelperObjectsABLWallFrictionVelocity {
  HelperObjectsABLWallFrictionVelocity(stk::mesh::BulkData& bulk, stk::mesh::Part* part, const double &gravity, const double &z0, const double &Tref)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    ABLWallFrictionAlgorithm(nullptr),
    gravity_(gravity),
    z0_(z0),
    Tref_(Tref)
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    ABLWallFrictionAlgorithm = new ComputeABLWallFrictionVelocityAlgorithm(realm, part, false, gravity_, z0_, Tref_);
  }

  ~HelperObjectsABLWallFrictionVelocity()
  {
    delete ABLWallFrictionAlgorithm;
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;

    delete naluObj;
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  unit_test_utils::NaluTest* naluObj;
  sierra::nalu::Realm& realm;
  ComputeABLWallFrictionVelocityAlgorithm* ABLWallFrictionAlgorithm;
  const double gravity_;
  const double z0_;
  const double Tref_;
};

struct HelperObjectsABLWallFunction {
  HelperObjectsABLWallFunction(stk::mesh::BulkData& bulk, int numDof, stk::mesh::Part* part, const double &z0, const double &Tref, const double &gravity)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    eqSystems(realm),
    eqSystem(eqSystems),
    linsys(new unit_test_utils::TestLinearSystem(realm, numDof, &eqSystem)),
    elemABLWallFunctionSolverAlg(nullptr),
    edgeABLWallFunctionSolverAlg(nullptr),
    computeGeomBoundAlg(nullptr),
    z0_(z0),
    Tref_(Tref),
    gravity_(gravity)
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    eqSystem.linsys_ = linsys;
    elemABLWallFunctionSolverAlg = new AssembleMomentumElemABLWallFunctionSolverAlgorithm(realm, part, &eqSystem, false, gravity_, z0_, Tref_);
    edgeABLWallFunctionSolverAlg = new AssembleMomentumEdgeABLWallFunctionSolverAlgorithm(realm, part, &eqSystem, gravity_, z0_, Tref_);
    computeGeomBoundAlg = new ComputeGeometryBoundaryAlgorithm(realm, part);
  }

  ~HelperObjectsABLWallFunction()
  {
    delete elemABLWallFunctionSolverAlg;
    delete edgeABLWallFunctionSolverAlg;
    delete computeGeomBoundAlg;
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;

    delete naluObj;
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  unit_test_utils::NaluTest* naluObj;
  sierra::nalu::Realm& realm;
  sierra::nalu::EquationSystems eqSystems;
  sierra::nalu::EquationSystem eqSystem;
  unit_test_utils::TestLinearSystem* linsys;
  AssembleMomentumElemABLWallFunctionSolverAlgorithm* elemABLWallFunctionSolverAlg;
  AssembleMomentumEdgeABLWallFunctionSolverAlgorithm* edgeABLWallFunctionSolverAlg;
  ComputeGeometryBoundaryAlgorithm* computeGeomBoundAlg;
  const double z0_;
  const double Tref_;
  const double gravity_;
};

#ifndef KOKKOS_ENABLE_CUDA

/*
  This test calls ABLWallFrictionVelocity::compute_utau() for three cases: neutral, unstable,
  and stable stratification.  The gold values for utau are obtained from a separate
  Matlab code implementation of the ABL friction velocity calculation.
*/
TEST(ABLWallFunction, compute_abl_utau) {

  stk::mesh::MetaData meta(3);
  stk::mesh::BulkData bulk(meta, MPI_COMM_WORLD);
  const double gravity = 9.81;
  const double z0 = 0.1;
  const double Tref = 300.0;
  HelperObjectsABLWallFrictionVelocity helperObjs(bulk, &meta.universal_part(), gravity, z0, Tref);

  const double tolerance = 1.0e-9;
  const double up = 1.563;
  const double zp  = 2.5;

  // Neutral
  const double qsurf_neutral = 0.0;
  NeutralABLProfileFunction NeutralProfFun;
  ABLProfileFunction *ABLProfFun = &NeutralProfFun;
  double utau;
  const double utau_neutral_gold = 0.199085033056820;

  helperObjs.ABLWallFrictionAlgorithm->compute_utau(up, zp, qsurf_neutral, ABLProfFun, utau);

  EXPECT_NEAR(utau, utau_neutral_gold, tolerance);

  // Unstable
  const double beta_m = 16.0;
  const double beta_h = 16.0;
  UnstableABLProfileFunction UnstableProfFun(beta_m, beta_h);
  ABLProfFun = &UnstableProfFun;
  const double qsurf_unstable = 0.281;
  const double utau_unstable_gold = 0.264845587455159;

  helperObjs.ABLWallFrictionAlgorithm->compute_utau(up, zp, qsurf_unstable, ABLProfFun, utau);

  EXPECT_NEAR(utau, utau_unstable_gold, tolerance);

  // Stable
  const double gamma_m = 5.0;
  const double gamma_h = 5.0;
  StableABLProfileFunction StableProfFun(gamma_m, gamma_h);
  ABLProfFun = &StableProfFun;
  const double qsurf_stable = -0.02;
  const double utau_stable_gold = 0.156653826868250;

  helperObjs.ABLWallFrictionAlgorithm->compute_utau(up, zp, qsurf_stable, ABLProfFun, utau);
  
  EXPECT_NEAR(utau, utau_stable_gold, tolerance);

}

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
  const double tolerance = 1.0e-6;
  const int numDof = 3;

  SetUp(rho_specified, utau_specified, up_specified, yp_specified);
  double rhs_gold = -rho_specified*utau_specified*utau_specified*aMag;
  HelperObjectsABLWallFunction helperObjs(bulk, numDof, &meta.universal_part(), z0, Tref, gravity);
  helperObjs.computeGeomBoundAlg->execute();

  // Element alg test
  helperObjs.elemABLWallFunctionSolverAlg->initialize_connectivity();
  helperObjs.elemABLWallFunctionSolverAlg->execute();

  unit_test_utils::TestLinearSystem *linsys = helperObjs.linsys;

  EXPECT_NEAR(linsys->rhs_(0), rhs_gold, tolerance);
}

/* This test creates and calls the ABL wall function edge algorithm
   for a single-element hex8 mesh and evaluates the resulting rhs vector
   for one of the faces against a pre-calculated value.
*/
TEST_F(ABLWallFunctionHex8ElementWithBCFields, abl_wall_function_edge_alg_rhs) {
  const double z0 = 0.1;
  const double Tref = 300.0;
  const double gravity = 9.81;
  const double rho_specified = 1.0;
  const double utau_specified = 0.067118435077841;
  const double up_specified = 0.15;
  const double yp_specified = 0.25;
  const double aMag = 0.25;
  const double tolerance = 1.0e-6;
  const int numDof = 3;

  SetUp(rho_specified, utau_specified, up_specified, yp_specified);
  double rhs_gold = -rho_specified*utau_specified*utau_specified*aMag;
  HelperObjectsABLWallFunction helperObjs(bulk, numDof, &meta.universal_part(), z0, Tref, gravity);
  helperObjs.computeGeomBoundAlg->execute();

  // Edge alg test
  helperObjs.elemABLWallFunctionSolverAlg->initialize_connectivity();
  helperObjs.realm.create_edges();
  helperObjs.edgeABLWallFunctionSolverAlg->execute();

  unit_test_utils::TestLinearSystem *linsys = helperObjs.linsys;

  EXPECT_NEAR(linsys->rhs_(0), rhs_gold, tolerance);
}

#endif

}
}
