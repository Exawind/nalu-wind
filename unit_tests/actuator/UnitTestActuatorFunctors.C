// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorFunctors.h>
#include <aero/actuator/ActuatorParsing.h>
#include <aero/actuator/ActuatorInfo.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <UnitTestUtils.h>
#include <yaml-cpp/yaml.h>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {
using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;

//-----------------------------------------------------------------

void
SetupActPoints(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorMemSpace> helper;
  ActVectorDbl point = actBulk.pointCentroid_.view_device();
  ActScalarDbl radius = actBulk.searchRadius_.view_device();

  helper.touch_dual_view(actBulk.pointCentroid_);
  helper.touch_dual_view(actBulk.searchRadius_);

  Kokkos::parallel_for(
    "SetupActPoints",
    Kokkos::RangePolicy<ActuatorExecutionSpace>(0, point.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      point(index, 0) = 1.0 + 1.5 * index;
      point(index, 1) = 2.5;
      point(index, 2) = 2.5;
      radius(index) = 2.0;
    });

  actBulk.searchRadius_.sync_host();
  actBulk.pointCentroid_.sync_host();
}

void
ComputeActuatorForce(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorMemSpace> helper;
  ActVectorDbl force = actBulk.actuatorForce_.view_device();
  ActVectorDbl velocity = actBulk.velocity_.view_device();

  helper.touch_dual_view(actBulk.actuatorForce_);
  helper.touch_dual_view(actBulk.velocity_);

  Kokkos::parallel_for(
    "CompActForce",
    Kokkos::RangePolicy<ActuatorExecutionSpace>(0, force.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      for (int j = 0; j < 3; j++) {
        force(index, j) = 1.2 * velocity(index, j);
      }
    });
  actBulk.actuatorForce_.sync_host();
  actBulk.velocity_.sync_host();
}

void
ActuatorTestInterpVelFunctors(
  const ActuatorMeta& actMeta,
  ActuatorBulk& actBulk,
  stk::mesh::BulkData& stkBulk)
{
  SetupActPoints(actBulk);

  actBulk.stk_search_act_pnts(actMeta, stkBulk);

  Kokkos::parallel_for(
    "interpVel", actMeta.numPointsTotal_, InterpActuatorVel(actBulk, stkBulk));

  auto vel = actBulk.velocity_.view_host();
  actuator_utils::reduce_view_on_host(vel);

  ComputeActuatorForce(actBulk);
}

struct FunctorTestSpread : public ActuatorBulk
{
  FunctorTestSpread(const ActuatorMeta& actMeta) : ActuatorBulk(actMeta) {}
};

void
InitSpreadTestFields(ActuatorBulk& actBulk)
{
  actBulk.epsilon_.modify_host();
  actBulk.searchRadius_.modify_host();
  actBulk.pointCentroid_.modify_host();
  actBulk.actuatorForce_.modify_host();

  auto epsilon = actBulk.epsilon_.view_host();
  auto radius = actBulk.searchRadius_.view_host();
  auto point = actBulk.pointCentroid_.view_host();
  auto force = actBulk.actuatorForce_.view_host();

  for (int i = 0; i < epsilon.extent_int(0); ++i) {
    for (int j = 0; j < 3; ++j) {
      epsilon(i, j) = 2.0;
      // assign at node to maximize overlap
      point(i, j) = 1.0 + i;
      force(i, j) = 1.0;
    }
    radius(i) = 2.0;
  }
}

struct ActuatorTestSpreadForceFunctor
{
  ActuatorTestSpreadForceFunctor(
    const ActuatorMeta& actMeta,
    ActuatorBulk& actBulk,
    stk::mesh::BulkData& stkBulk)
    : actMeta_(actMeta),
      actBulk_(actBulk),
      stkBulk_(stkBulk),
      numActPoints_(actMeta_.numPointsTotal_)
  {
  }

  void operator()()
  {

    InitSpreadTestFields(actBulk_);

    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);
    const int localSizeCoarseSearch =
      actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

    Kokkos::parallel_for(
      "spreadForce", localSizeCoarseSearch,
      SpreadActuatorForce(actBulk_, stkBulk_));
  }

  const ActuatorMeta& actMeta_;
  ActuatorBulk& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
};

namespace {
//-----------------------------------------------------------------
class ActuatorFunctorTests : public ::testing::Test
{
protected:
  std::string inputFileSurrogate_;
  stk::mesh::MetaData* stkMeta_;
  std::shared_ptr<stk::mesh::BulkData> stkBulk_;
  const double tol_;
  const VectorFieldType* coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  VectorFieldType* actuatorForce_{nullptr};
  ScalarFieldType* dualNodalVolume_{nullptr};

  ActuatorFunctorTests() : tol_(1e-8), coordinates_(nullptr)
  {
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(3);
    stkBulk_ = meshBuilder.create();
    stkMeta_ = &stkBulk_->mesh_meta_data();

    velocity_ = &stkMeta_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");
    actuatorForce_ = &stkMeta_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source");
    dualNodalVolume_ = &stkMeta_->declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume");

    stk::mesh::put_field_on_mesh(
      *velocity_, stkMeta_->universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(
      *actuatorForce_, stkMeta_->universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(
      *dualNodalVolume_, stkMeta_->universal_part(), 1, nullptr);
    stk::mesh::field_fill(1.0, *dualNodalVolume_);
  }

  void SetUp()
  {
    const std::string meshSpec = "generated:5x5x5";
    unit_test_utils::fill_hex8_mesh(meshSpec, *stkBulk_);
    coordinates_ =
      static_cast<const VectorFieldType*>(stkMeta_->coordinate_field());
    const stk::mesh::Selector selector =
      stkMeta_->locally_owned_part() | stkMeta_->globally_shared_part();
    const auto& buckets =
      stkBulk_->get_buckets(stk::topology::NODE_RANK, selector);
    for (const stk::mesh::Bucket* bptr : buckets) {
      for (stk::mesh::Entity node : *bptr) {
        const double* coords = stk::mesh::field_data(*coordinates_, node);
        double* vel = stk::mesh::field_data(*velocity_, node);
        double* aF = stk::mesh::field_data(*actuatorForce_, node);
        for (int i = 0; i < 3; i++) {
          vel[i] = coords[i];
          aF[i] = 0.0;
        }
      }
    }
  }
};

TEST_F(ActuatorFunctorTests, NGP_testSearchAndInterpolate)
{
  inputFileSurrogate_ = "actuator:\n"
                        "  type: ActLinePointDrag\n"
                        "  n_turbines_glob: 1\n"
                        "  search_method: stk_kdtree\n"
                        "  search_target_part: [block_1]\n"
                        "  Turbine0:\n"
                        "    num_force_pts_blade: 3";
  YAML::Node y_actuator = YAML::Load(inputFileSurrogate_);
  ActuatorMeta actMeta = actuator_parse(y_actuator);
  actMeta.numPointsTotal_ = 3;

  // construct object and allocate memory
  ActuatorBulk actBulk(actMeta);

  // what gets called in the time loop
  ActuatorTestInterpVelFunctors(actMeta, actBulk, *stkBulk_);

  // check results
  const int nTotal = actMeta.numPointsTotal_;
  auto points = actBulk.pointCentroid_.template view<Kokkos::HostSpace>();
  auto vel = actBulk.velocity_.template view<Kokkos::HostSpace>();
  auto force = actBulk.actuatorForce_.template view<Kokkos::HostSpace>();
  for (int i = 0; i < nTotal; i++) {
    EXPECT_NEAR(1.0 + 1.5 * i, vel(i, 0), tol_);
    EXPECT_NEAR(2.5, vel(i, 1), tol_);
    EXPECT_NEAR(2.5, vel(i, 2), tol_);
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(vel(i, j) * 1.2, force(i, j), tol_);
    }
  }
}

TEST_F(ActuatorFunctorTests, NGP_testSpreadForces)
{
  inputFileSurrogate_ = "actuator:\n"
                        "  type: ActLinePointDrag\n"
                        "  n_turbines_glob: 1\n"
                        "  search_method: stk_kdtree\n"
                        "  search_target_part: [block_1]\n"
                        "  Turbine0:\n"
                        "    num_force_pts_blade: 1";
  YAML::Node y_actuator = YAML::Load(inputFileSurrogate_);
  ActuatorMeta actMeta = actuator_parse(y_actuator);
  actMeta.numPointsTotal_ = 1;

  ActuatorInfoNGP actInfo;
  actInfo.epsilon_.x_ = 2.0;
  actInfo.epsilon_.y_ = 2.0;
  actInfo.epsilon_.z_ = 2.0;
  actMeta.add_turbine(actInfo);

  ActuatorBulk actBulk(actMeta);
  ActuatorTestSpreadForceFunctor(actMeta, actBulk, *stkBulk_)();

  auto coarseElems = actBulk.coarseSearchElemIds_.view_host();
  const int numCoarse = coarseElems.extent_int(0);

  // make sure local search results get non-zero source term
  std::vector<stk::mesh::Entity> nodesMatch;

  for (int i = 0; i < numCoarse; ++i) {
    const stk::mesh::Entity elem =
      stkBulk_->get_entity(stk::topology::ELEMENT_RANK, coarseElems(i));
    stk::mesh::Entity const* elem_node_rels = stkBulk_->begin_nodes(elem);
    const unsigned numNodes = stkBulk_->num_nodes(elem);
    for (unsigned j = 0; j < numNodes; ++j) {
      stk::mesh::Entity node = elem_node_rels[j];
      nodesMatch.push_back(node);
      double* actSource = (double*)stk::mesh::field_data(*actuatorForce_, node);
      for (int k = 0; k < 3; ++k) {
        EXPECT_TRUE(actSource[k] > 0.0)
          << "Value is: " << actSource[k] << std::endl
          << "Elem is: " << coarseElems(i) << std::endl
          << "Node is: " << j << std::endl
          << "Index is: " << k;
      }
    }
  }

  // make sure all nodes are now zero
  const stk::mesh::Selector selector =
    stkMeta_->locally_owned_part() | stkMeta_->globally_shared_part();
  const auto& buckets =
    stkBulk_->get_buckets(stk::topology::NODE_RANK, selector);
  for (const stk::mesh::Bucket* bptr : buckets) {
    for (stk::mesh::Entity node : *bptr) {
      if (
        std::find(nodesMatch.begin(), nodesMatch.end(), node) ==
        nodesMatch.end()) {
        const double* aF = stk::mesh::field_data(*actuatorForce_, node);
        for (int i = 0; i < 3; i++) {
          EXPECT_DOUBLE_EQ(aF[i], 0.0);
        }
      }
    }
  }
}

} // namespace

} /* namespace nalu */
} /* namespace sierra */
