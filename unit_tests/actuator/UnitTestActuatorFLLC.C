// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details

#include <gtest/gtest.h>
#include <aero/actuator/ActuatorTypes.h>
#include <aero/actuator/ActuatorParsing.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#include <aero/actuator/ActuatorParsingSimple.h>
#include <aero/actuator/ActuatorFunctorsSimple.h>
#include <aero/actuator/ActuatorFLLC.h>
#include <yaml-cpp/yaml.h>

namespace sierra {
namespace nalu {

namespace {
const char* actuatorParameters = R"act(actuator:
  search_target_part: dummy
  search_method: stk_kdtree
  type: ActLineSimpleNGP
  n_simpleblades: 1
  Blade0:
    fllt_correction: yes
    num_force_pts_blade: 5
    epsilon_min: [3.0, 3.0, 3.0]
    epsilon_chord: [0.25, 0.25, 0.25]
    p1: [0, -4, 0]
    p2: [0,  4, 0]
    p1_zero_alpha_dir: [1, 0, 0]
    chord_table: [1.0]
    twist_table: [0.0]
    aoa_table: [-180, 0, 180]
    cl_table:  [2, 2, 2]
    cd_table:  [1.2])act";
class ActuatorFLLC : public ::testing::Test
{

public:
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActuatorMeta actMetaBase_;
  ActuatorMetaSimple actMeta_;
  ActuatorBulkSimple actBulk_;

  ActuatorFLLC()
    : actMetaBase_(actuator_parse(YAML::Load(actuatorParameters))),
      actMeta_(
        actuator_Simple_parse(YAML::Load(actuatorParameters), actMetaBase_)),
      actBulk_(actMeta_)
  {
  }

  void SetUp()
  {
    ASSERT_TRUE(actMetaBase_.useFLLC_);
    ASSERT_TRUE(actMeta_.useFLLC_);
    ASSERT_EQ(actMetaBase_.actuatorType_, ActuatorType::ActLineSimpleNGP);
    ASSERT_EQ(actMeta_.actuatorType_, ActuatorType::ActLineSimpleNGP);
  }
};

TEST_F(ActuatorFLLC, NGP_ComputeLiftForceDistribution_G_Eq_5_3)
{
  auto vel = helper_.get_local_view(actBulk_.velocity_);
  auto relVel = helper_.get_local_view(actBulk_.relativeVelocity_);
  auto density = helper_.get_local_view(actBulk_.density_);
  auto spanDir = helper_.get_local_view(actMeta_.spanDir_);

  auto range_policy = actBulk_.local_range_policy();
  Kokkos::parallel_for(
    "init velocities", range_policy, ACTUATOR_LAMBDA(int i) {
      for (int j = 0; j < 3; ++j) {
        vel(i, j) = 1.0;
      }
      density(i) = 2.0;
    });

  ActSimpleComputeRelativeVelocity(actBulk_, actMeta_);
  ActSimpleComputeForce(actBulk_, actMeta_);

  // given a CL, CD, chord and U vector we can compute a lift and total force
  // then ensure our computation gives the expected lift force distribution from
  // the paper

  ActFixScalarDbl G("G-paper", vel.extent_int(0));

  const double chord = 1.0;
  const double Cl = 2.0;

  auto area = helper_.get_local_view(actMeta_.elemAreaDv_);
  auto dR = helper_.get_local_view(actMeta_.dR_);

  Kokkos::parallel_for(
    "compute G like paper", range_policy, ACTUATOR_LAMBDA(int i) {
      const double umag2 = relVel(i, 0) * relVel(i, 0) +
                           relVel(i, 1) * relVel(i, 1) +
                           relVel(i, 2) * relVel(i, 2);
      ASSERT_DOUBLE_EQ(chord, area(0, i) / dR(0));
      G(i) = 0.5 * chord * Cl * umag2; // chord is 1.0 and Cl is 2.0 everywhere
    });

  actuator_utils::reduce_view_on_host(G);
  FilteredLiftingLineCorrection fllc(actMeta_, actBulk_);
  fllc.compute_lift_force_distribution();

  auto fllc_lift_force =
    helper_.get_local_view(actBulk_.liftForceDistribution_);
  // assert that the two lift forces are equal
  Kokkos::parallel_for(
    "check values", range_policy, ACTUATOR_LAMBDA(int i) {
      double gmag = 0.0;
      for (int j = 0; j < 3; ++j) {
        gmag += fllc_lift_force(i, j) * fllc_lift_force(i, j);
      }
      gmag = std::sqrt(gmag);
      EXPECT_DOUBLE_EQ(G(i), gmag);
    });
}

TEST_F(ActuatorFLLC, NGP_ComputeGradG_Eq_5_4_and_5_5)
{
  auto G = helper_.get_local_view(actBulk_.liftForceDistribution_);
  auto points = helper_.get_local_view(actBulk_.pointCentroid_);

  ASSERT_TRUE(points.extent_int(0) > 2);

  const double fixedDR[3] = {0.5, 0.1, 0.6};

  auto range_policy = actBulk_.local_range_policy();
  ActFixVectorDbl r("radius", G.extent_int(0));
  // create a parabola from the point locations then compute deltaG and dG/dr
  Kokkos::parallel_for(
    "init G as r^2", range_policy, ACTUATOR_LAMBDA(int i) {
      for (int j = 0; j < 3; ++j) {
        r(i, j) = i * fixedDR[j];
        G(i, j) = r(i, j) * r(i, j);
      }
    });
  actuator_utils::reduce_view_on_host(G);
  actuator_utils::reduce_view_on_host(r);

  FilteredLiftingLineCorrection fllc(actMeta_, actBulk_);
  fllc.grad_lift_force_distribution();

  auto dG = helper_.get_local_view(actBulk_.deltaLiftForceDistribution_);

  const int lastPoint = G.extent_int(0) - 1;

  for (int i = 1; i < lastPoint; ++i) {
    for (int j = 0; j < 3; ++j)
      // compare against analytical derivative
      EXPECT_NEAR(2.0 * r(i, j), dG(i, j) / fixedDR[j], 1e-12)
        << "index: " << i << ", " << j << " radius: " << r(i, j)
        << "  G(i+1): " << G(i + 1, j) << " G(i-1): " << G(i - 1, j);
  }
  // Check eq 5.5 a and b
  for (int j = 0; j < 3; ++j) {
    EXPECT_DOUBLE_EQ(G(0, j), dG(0, j));
    EXPECT_DOUBLE_EQ(-G(lastPoint, j), dG(lastPoint, j));
  }
}

/*
 Equaiton 5.7  from the paper should be
 (note Uinf is incorrect in the paper and should be indexed on j inside the
 summation):

 u_y(z_i, eps_i) =  \sum_j \Delta G(z_j) / (-Uinf_j*4* \pi *
 (r_ij))*(1-exp(-r_ij^2/eps_i^2))

 where r_ij = z_i-z_j = dr*(i-j) (for fixed point spacing)

 if we set

 \Delta G(z_j) = 4\pi
 dr = 1.0
 eps = 1.0/sqrt(ln(\xi)) where \xi >1.0
 Uinf_j = 1.0

 the equaiton reduces to

 \delta U_i =\sum_j -1/(i-j) * (1-\xi^(-(i-j)^2))

 because \Delta G will cancel the 4 \pi term
 and the magnitude of the relative velocity is fixed to dr

 so for correction (old timestep values are zero in this test)
 f*(\delta U_opt_i- \delta U_les_i)

 the 1 minus tersm will cancel out leaving just the exponential terms
 which we can wrap into one summation as

 f*(\sum_j [\xi_opt^(-(i-j)^2) - \xi_les^(-(i-j)^2)]/(i-j) )

 this is the equaiton we will test against to verify the computation
*/
TEST_F(ActuatorFLLC, NGP_ComputeInducedVelocity_Eq_5_7)
{
  auto Uinf = helper_.get_local_view(actBulk_.relativeVelocityMagnitude_);
  auto dG = helper_.get_local_view(actBulk_.deltaLiftForceDistribution_);
  auto epsLES = helper_.get_local_view(actBulk_.epsilon_);
  auto epsOpt = helper_.get_local_view(actBulk_.epsilonOpt_);
  auto points = helper_.get_local_view(actBulk_.pointCentroid_);
  auto uInduced = helper_.get_local_view(actBulk_.fllc_);

  auto range_policy = actBulk_.local_range_policy();

  const double lesFac = 3.0;
  const double optFac = 2.0;
  const double epsilonLES = 1.0 / std::sqrt(std::log(lesFac));
  const double epsilonOpt = 1.0 / std::sqrt(std::log(optFac));

  ActFixVectorDbl uExpect("uExpect", dG.extent_int(0));
  const int offset =
    helper_.get_local_view(actBulk_.turbIdOffset_)(actBulk_.localTurbineId_);
  const int numPoints = helper_.get_local_view(actMeta_.numPointsTurbine_)(
    actBulk_.localTurbineId_);

  helper_.touch_dual_view(actBulk_.epsilonOpt_);
  helper_.touch_dual_view(actBulk_.epsilon_);
  helper_.touch_dual_view(actBulk_.pointCentroid_);
  Kokkos::deep_copy(epsOpt, epsilonOpt);
  Kokkos::deep_copy(epsLES, epsilonLES);
  Kokkos::deep_copy(points, 0.0);
  Kokkos::deep_copy(dG, 4.0 * M_PI);
  Kokkos::deep_copy(Uinf, 1.0);

  Kokkos::parallel_for(
    "init values", range_policy, ACTUATOR_LAMBDA(int index) {
      points(index, 0) = index;
      points(index, 1) = 0.0;
      points(index, 2) = 0.0;
    });

  actuator_utils::reduce_view_on_host(points);

  Kokkos::parallel_for(
    "compute values", range_policy, ACTUATOR_LAMBDA(int index) {
      const int i = index - offset;
      for (int j = 0; j < numPoints; ++j) {
        if (i == j)
          continue;

        const double r = (i - j);
        const double r2 = r * r;
        double temp = (std::pow(optFac, -r2) - std::pow(lesFac, -r2)) / r;

        for (int k = 0; k < 3; ++k) {
          uExpect(index, k) -= 0.1 * temp;
        }
      }
    });

  for (int i = 0; i < numPoints; ++i) {
    EXPECT_DOUBLE_EQ(epsOpt(i, 0), epsilonOpt) << epsOpt(i, 0);
  }

  FilteredLiftingLineCorrection fllc(actMeta_, actBulk_);
  fllc.compute_induced_velocities();
  actuator_utils::reduce_view_on_host(uExpect);

  for (int i = 0; i < uExpect.extent_int(0); ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(uExpect(i, j), uInduced(i, j), 1e-12) << "index: " << i;
    }
  }
}

} // namespace
} // namespace nalu
} // namespace sierra
