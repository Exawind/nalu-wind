// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFunctors.h>
#include <actuator/ActuatorParsing.h>
#include <actuator/ActuatorInfo.h>
#include <actuator/UtilitiesActuator.h>
#include <UnitTestUtils.h>
#include <yaml-cpp/yaml.h>
#include <gtest/gtest.h>

namespace sierra
{
namespace nalu
{
using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;

//-----------------------------------------------------------------
struct SetPoints{};
struct ComputeForce{};
struct Interpolate{};


using SetupActPoints = ActuatorFunctor<
  ActuatorBulk,
  SetPoints,
  ActuatorExecutionSpace>;
template <>
SetupActPoints::ActuatorFunctor(ActuatorBulk& actBulk)
  : actBulk_(actBulk)
{
  touch_dual_view(actBulk_.pointCentroid_);
  touch_dual_view(actBulk_.searchRadius_);
}

template <>
void
SetupActPoints::operator()(const int& index) const
{
  auto point = get_local_view(actBulk_.pointCentroid_);
  auto radius = get_local_view(actBulk_.searchRadius_);
  point(index, 0) = 1.0 + 1.5 * index;
  point(index, 1) = 2.5;
  point(index, 2) = 2.5;
  radius(index) = 2.0;
}

using ComputeActuatorForce = ActuatorFunctor<
  ActuatorBulk,
  ComputeForce,
  ActuatorExecutionSpace>;
template <>
ComputeActuatorForce::ActuatorFunctor(ActuatorBulk& actBulk)
  : actBulk_(actBulk)
{
  touch_dual_view(actBulk_.actuatorForce_);
}

template <>
void
ComputeActuatorForce::operator()(const int& index) const
{
  auto force = get_local_view(actBulk_.actuatorForce_);
  auto velocity = get_local_view(actBulk_.velocity_);
  for (int j = 0; j < 3; j++) {
    force(index, j) = 1.2 * velocity(index, j);
  }
}


struct ActuatorTestInterpVelFunctors{
  ActuatorTestInterpVelFunctors(const ActuatorMeta& actMeta,
    ActuatorBulk& actBulk,
    stk::mesh::BulkData& stkBulk):
      actMeta_(actMeta),
      actBulk_(actBulk),
      stkBulk_(stkBulk),
      numActPoints_(actBulk_.totalNumPoints_)
  {}

  void operator()()
  {

    Kokkos::parallel_for(
      "setPointLocations", numActPoints_, SetupActPoints(actBulk_));

    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);

    Kokkos::parallel_for("interpVel", numActPoints_, InterpActuatorVel(actBulk_, stkBulk_));

    auto vel = actBulk_.velocity_.view_host();
    actuator_utils::reduce_view_on_host(vel);

    Kokkos::parallel_for(
      "computeActuatorForce", numActPoints_,
      ComputeActuatorForce(actBulk_));
  }

  const ActuatorMeta& actMeta_;
  ActuatorBulk& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
};


struct FunctorTestSpread : public ActuatorBulk{
  FunctorTestSpread(const ActuatorMeta& actMeta):
    ActuatorBulk(actMeta){}
};

void InitSpreadTestFields(ActuatorBulk& actBulk){
  actBulk.epsilon_.modify_host();
  actBulk.searchRadius_.modify_host();
  actBulk.pointCentroid_.modify_host();
  actBulk.actuatorForce_.modify_host();

  auto epsilon = actBulk.epsilon_.view_host();
  auto radius = actBulk.searchRadius_.view_host();
  auto point = actBulk.pointCentroid_.view_host();
  auto force = actBulk.actuatorForce_.view_host();

  for(int i=0; i<epsilon.extent_int(0); ++i){
    for(int j=0; j<3; ++j){
      epsilon(i,j)=2.0;
      // assign at node to maximize overlap
      point(i,j) = 1.0 + i;
      force(i,j) = 1.0;
    }
    radius(i) = 2.0;
  }
}

struct ActuatorTestSpreadForceFunctor{
  ActuatorTestSpreadForceFunctor(const ActuatorMeta& actMeta,
    ActuatorBulk& actBulk,
    stk::mesh::BulkData& stkBulk):
      actMeta_(actMeta),
      actBulk_(actBulk),
      stkBulk_(stkBulk),
      numActPoints_(actBulk_.totalNumPoints_)
  {}

  void operator()()
  {

    InitSpreadTestFields(actBulk_);

    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);
    const int localSizeCoarseSearch = actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

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
  stk::mesh::MetaData stkMeta_;
  stk::mesh::BulkData stkBulk_;
  const double tol_;
  const VectorFieldType* coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  VectorFieldType* actuatorForce_{nullptr};

  ActuatorFunctorTests()
    : stkMeta_(3),
      stkBulk_(stkMeta_, MPI_COMM_WORLD),
      tol_(1e-8),
      coordinates_(nullptr),
      velocity_(&stkMeta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "velocity")),
      actuatorForce_(&stkMeta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "actuator_source"))
  {
    stk::mesh::put_field_on_mesh(
      *velocity_, stkMeta_.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(
      *actuatorForce_, stkMeta_.universal_part(), 3, nullptr);
  }

  void SetUp()
  {
    const std::string meshSpec = "generated:5x5x5";
    unit_test_utils::fill_hex8_mesh(meshSpec, stkBulk_);
    coordinates_ =
      static_cast<const VectorFieldType*>(stkMeta_.coordinate_field());
    const stk::mesh::Selector selector =
      stkMeta_.locally_owned_part() | stkMeta_.globally_shared_part();
    const auto& buckets =
      stkBulk_.get_buckets(stk::topology::NODE_RANK, selector);
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

TEST_F(ActuatorFunctorTests, testSearchAndInterpolate)
{
  inputFileSurrogate_ = "actuator:\n"
                        "  type: ActLinePointDrag\n"
                        "  n_turbines_glob: 1\n"
                        "  search_method: stk_kdtree\n"
                        "  search_target_part: [block_1]\n";
  YAML::Node y_actuator = YAML::Load(inputFileSurrogate_);
  ActuatorMeta actMeta = actuator_parse(y_actuator);

  // more parse stuff to be implemented
  ActuatorInfoNGP actInfo;
  actInfo.numPoints_ = 3;
  actMeta.add_turbine(actInfo);

  // construct object and allocate memory
  ActuatorBulk actBulk(actMeta);

  // what gets called in the time loop
  ActuatorTestInterpVelFunctors(actMeta, actBulk, stkBulk_)();

  // check results
  const int nTotal = actBulk.totalNumPoints_;
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

TEST_F(ActuatorFunctorTests, testSpreadForces){
  inputFileSurrogate_ = "actuator:\n"
                        "  type: ActLinePointDrag\n"
                        "  n_turbines_glob: 1\n"
                        "  search_method: stk_kdtree\n"
                        "  search_target_part: [block_1]\n";
  YAML::Node y_actuator = YAML::Load(inputFileSurrogate_);
  ActuatorMeta actMeta = actuator_parse(y_actuator);

  ActuatorInfoNGP actInfo;
  actInfo.numPoints_ = 1;
  actInfo.epsilon_.x_=2.0;
  actInfo.epsilon_.y_=2.0;
  actInfo.epsilon_.z_=2.0;
  actMeta.add_turbine(actInfo);

  ActuatorBulk actBulk(actMeta);
  ActuatorTestSpreadForceFunctor(actMeta, actBulk, stkBulk_)();

  auto coarseElems = actBulk.coarseSearchElemIds_.view_host();
  const int numCoarse = coarseElems.extent_int(0);

  //make sure local search results get non-zero source term
  std::vector<stk::mesh::Entity> nodesMatch;

  for(int i=0; i<numCoarse; ++i){
    const stk::mesh::Entity elem = stkBulk_.get_entity(stk::topology::ELEMENT_RANK, coarseElems(i));
    stk::mesh::Entity const* elem_node_rels = stkBulk_.begin_nodes(elem);
    const unsigned numNodes = stkBulk_.num_nodes(elem);
    for (unsigned j =0; j<numNodes; ++j){
      stk::mesh::Entity node = elem_node_rels[j];
      nodesMatch.push_back(node);
      double* actSource = (double*) stk::mesh::field_data(*actuatorForce_,node);
      for(int k=0; k<3; ++k){
        EXPECT_TRUE(actSource[k]>0.0)
            <<"Value is: " << actSource[k] <<std::endl
            <<"Elem is: "<<coarseElems(i)<<std::endl
            <<"Node is: "<<j<<std::endl
            <<"Index is: "<<k;
      }
    }
  }

  //make sure all nodes are now zero
  const stk::mesh::Selector selector =
    stkMeta_.locally_owned_part() | stkMeta_.globally_shared_part();
  const auto& buckets =
    stkBulk_.get_buckets(stk::topology::NODE_RANK, selector);
  for (const stk::mesh::Bucket* bptr : buckets) {
    for (stk::mesh::Entity node : *bptr) {
      if(std::find(nodesMatch.begin(), nodesMatch.end(),node)==nodesMatch.end()){
        const double* aF = stk::mesh::field_data(*actuatorForce_, node);
        for (int i = 0; i < 3; i++) {
          EXPECT_DOUBLE_EQ(aF[i],0.0);
        }
      }
    }
  }
}

}

} /* namespace nalu */
} /* namespace sierra */
