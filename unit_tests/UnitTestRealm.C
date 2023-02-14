// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include "NaluEnv.h"
#include "LinearSolvers.h"
#include "Realms.h"
#include "Realm.h"
#include "InputOutputRealm.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "ComputeSSTMaxLengthScaleElemAlgorithm.h"

#include <string>

namespace {

const std::string naluDefaultInputs =
  "Simulations:                                                            \n"
  "  - name: sim1                                                          \n"
  "    time_integrator: ti_1                                               \n"
  "    optimizer: opt1                                                     \n"
  "                                                                        \n"
  "linear_solvers:                                                         \n"
  "                                                                        \n"
#ifdef NALU_USES_TRILINOS_SOLVERS
  "  - name: solve_scalar                                                  \n"
  "    type: tpetra                                                        \n"
  "    method: gmres                                                       \n"
  "    preconditioner: sgs                                                 \n"
  "    tolerance: 1e-5                                                     \n"
  "    max_iterations: 50                                                  \n"
  "    kspace: 50                                                          \n"
  "    output_level: 0                                                     \n"
  "                                                                        \n"
  "  - name: solve_cont                                                    \n"
  "    type: tpetra                                                        \n"
  "    method: gmres                                                       \n"
  "    preconditioner: muelu                                               \n"
  "    tolerance: 1e-5                                                     \n"
  "    max_iterations: 50                                                  \n"
  "    kspace: 50                                                          \n"
  "    output_level: 0                                                     \n"
  "    recompute_preconditioner: no                                        \n"
  "    muelu_xml_file_name: milestone.xml                                  \n"
#else // NALU_USES_HYPRE
  "  - name: solve_scalar                                                  \n"
  "    type: hypre                                                         \n"
  "    method: hypre_gmres                                                 \n"
  "    preconditioner: boomerAMG                                           \n"
  "    tolerance: 1e-5                                                     \n"
  "    max_iterations: 50                                                  \n"
  "    kspace: 50                                                          \n"
  "    output_level: 0                                                     \n"
  "                                                                        \n"
  "  - name: solve_cont                                                    \n"
  "    type: hypre                                                         \n"
  "    method: hypre_gmres                                                 \n"
  "    preconditioner: boomerAMG                                           \n"
  "    tolerance: 1e-5                                                     \n"
  "    max_iterations: 50                                                  \n"
  "    kspace: 50                                                          \n"
  "    output_level: 0                                                     \n"
  "    recompute_preconditioner: no                                        \n"
#endif
  "                                                                        \n"
  "Time_Integrators:                                                       \n"
  "  - StandardTimeIntegrator:                                             \n"
  "      name: ti_1                                                        \n"
  "      start_time: 0                                                     \n"
  "      time_step: 0.1                                                    \n"
  "      termination_time: 1.5                                             \n"
  "      time_stepping_type: adaptive                                      \n"
  "      time_step_count: 0                                                \n"
  "      second_order_accuracy: yes                                        \n"
  "                                                                        \n"
  "      realms: []                                                        \n"
  "                                                                        \n";

const std::string realmDefaultSettings =
  "- name: unitTestRealm                                                  \n"
  "  use_edges: yes                                                       \n"
  "                                                                       \n"
  "  equation_systems:                                                    \n"
  "    name: theEqSys                                                     \n"
  "    max_iterations: 2                                                  \n"
  "                                                                       \n"
  "    solver_system_specification:                                       \n"
  "      ndtw : solve_scalar                                              \n"
  "                                                                       \n"
  "    systems:                                                           \n"
  "      - WallDistance:                                                  \n"
  "          name: myNDTW                                                 \n"
  "          max_iterations: 1                                            \n"
  "          convergence_tolerance: 1e-5                                  \n"
  "                                                                       \n"
  "  time_step_control:                                                   \n"
  "    target_courant: 2.0                                                \n"
  "    time_step_change_factor: 1.2                                       \n"
  "                                                                       \n"
  "  solution_options:                                                    \n"
  "    name: unitTestRealmOptions                                         \n"
  "    turbulence_model: laminar                                          \n"
  "    interp_rhou_together_for_mdot: yes                                 \n"
  "    reduced_sens_cvfem_poisson: no                                     \n"
  "                                                                       \n"
  "    options:                                                           \n"
  "      - laminar_prandtl:                                               \n"
  "          enthalpy: 0.7                                                \n"
  "      - turbulent_prandtl:                                             \n"
  "          enthalpy: 1.0                                                \n"
  "      - shifted_gradient_operator:                                     \n"
  "          velocity: no                                                 \n"
  "          pressure: no                                                 \n"
  "          mixture_fraction: no                                         \n";
} // namespace

namespace unit_test_utils {

YAML::Node
get_default_inputs()
{
  YAML::Node doc = YAML::Load(naluDefaultInputs);

  return doc;
}

YAML::Node
get_realm_default_node()
{
  YAML::Node node = YAML::Load(realmDefaultSettings);
  return node[0];
}

NaluTest::NaluTest(const YAML::Node& doc)
  : comm_(MPI_COMM_WORLD),
    spatialDim_(3),
    sim_(doc),
    logFileName_("unittestX_naluwrapper.log")
{
  // NaluEnv log file
  auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  if (testInfo) {
    std::string caseName = testInfo->test_case_name();
    std::string caseInstance = testInfo->name();
    logFileName_ = caseName + "." + caseInstance + ".log";
  }
  sierra::nalu::NaluEnv::self().set_log_file_stream(logFileName_, false);

  sim_.linearSolvers_ = new sierra::nalu::LinearSolvers(sim_);
  sim_.realms_ = new sierra::nalu::Realms(sim_);
  sim_.timeIntegrator_ = new sierra::nalu::TimeIntegrator(&sim_);

  sim_.linearSolvers_->load(doc);
  sim_.timeIntegrator_->load(doc);
}

NaluTest::~NaluTest() { unlink(logFileName_.c_str()); }

sierra::nalu::Realm&
NaluTest::create_realm(
  const YAML::Node& realm_node,
  const std::string realm_type,
  const bool createMeshObjects)
{
  sierra::nalu::Realm* realm = nullptr;
  if (realm_type == "multi_physics") {
    realm = new sierra::nalu::Realm(*sim_.realms_, realm_node);
    realm->solutionOptions_->load(realm_node);
    realm->equationSystems_.load(realm_node);
  } else {
    realm = new sierra::nalu::InputOutputRealm(*sim_.realms_, realm_node);
    realm->solutionOptions_->load(realm_node);
  }

  // Set-up mesh metadata and bulkdata ... let user fill mesh in test function
  if (createMeshObjects) {
    stk::mesh::MeshBuilder meshBuilder(comm_);
    meshBuilder.set_spatial_dimension(spatialDim_);
    realm->bulkData_ = meshBuilder.create();
  }
  sim_.realms_->realmVector_.push_back(realm);

  return *realm;
}

void
verify_field_values(
  double expectedValue,
  ScalarFieldType* maxLengthScaleField,
  const stk::mesh::BulkData& mesh)
{
  const stk::mesh::BucketVector& nodeBuckets =
    mesh.buckets(stk::topology::NODE_RANK);
  EXPECT_FALSE(nodeBuckets.empty());
  for (const stk::mesh::Bucket* bucketPtr : nodeBuckets) {
    if (!bucketPtr->owned()) {
      continue;
    }
    for (stk::mesh::Entity node : *bucketPtr) {
      double* fieldValues =
        static_cast<double*>(stk::mesh::field_data(*maxLengthScaleField, node));
      EXPECT_NEAR(expectedValue, fieldValues[0], 1.e-8)
        << "at node " << mesh.entity_key(node);
    }
  }
}

TEST(NaluMock, test_nalu_mock)
{
  // 1. Create dummy YAML inputs (mimics an input file)
  YAML::Node doc = get_default_inputs();
  unit_test_utils::NaluTest naluObj(doc);

  // 2. Create a Realm input node used to fill data
  const YAML::Node realm_node = get_realm_default_node();
  // Modify the default node, if necessary, for the test

  // 3. Create the Realm
  sierra::nalu::Realm& realm = naluObj.create_realm(realm_node);

  // 4. Create necessary fields...
  ScalarFieldType& maxLengthScaleField =
    realm.meta_data().declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "sst_max_length_scale");
  double zero = 0.0;
  stk::mesh::put_field_on_mesh(
    maxLengthScaleField, realm.meta_data().universal_part(), &zero);

  // 5. Create mesh and get the default part for registration with Algorithm
  unit_test_utils::fill_hex8_mesh("generated:10x10x10", realm.bulk_data());
  stk::mesh::Part* meshPart = realm.meta_data().get_part("block_1");

  // 6. Initialize the Algorithm to be tested...
  sierra::nalu::ComputeSSTMaxLengthScaleElemAlgorithm sstAlg(realm, meshPart);

  // 7. Perform tests
  EXPECT_TRUE(sstAlg.coordinates_ != nullptr);
  EXPECT_TRUE(sstAlg.maxLengthScale_ != nullptr);

  sstAlg.execute();

  // for our generated-mesh case, we expect the maxLengthScale_ field to be all
  // ones...
  double expectedValue = 1.0;
  verify_field_values(expectedValue, &maxLengthScaleField, realm.bulk_data());
}

} // namespace unit_test_utils
