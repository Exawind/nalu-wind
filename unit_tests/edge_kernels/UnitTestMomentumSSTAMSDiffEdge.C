// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/MomentumSSTAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumSSTLRAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumKOAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumKEAMSDiffEdgeKernel.h"

namespace {

const std::string realmSSTAMSSettings =
  "- name: unitTestRealm                                                  \n"
  "  use_edges: yes                                                       \n"
  "                                                                       \n"
  "  equation_systems:                                                    \n"
  "    name: theEqSys                                                     \n"
  "    max_iterations: 2                                                  \n"
  "                                                                       \n"
  "    solver_system_specification:                                       \n"
  "      velocity: solve_scalar                                           \n"
  "      turbulent_ke: solve_scalar                                       \n"
  "      specific_dissipation_rate: solve_scalar                          \n"
  "      pressure: solve_cont                                             \n"
  "      ndtw: solve_cont                                                 \n"
  "                                                                       \n"
  "    systems:                                                           \n"
  "      - WallDistance:                                                  \n"
  "          name: myNDTW                                                 \n"
  "          max_iterations: 1                                            \n"
  "          convergence_tolerance: 1e-5                                  \n"
  "                                                                       \n"
  "      - LowMachEOM:                                                    \n"
  "          name: myLowMach                                              \n"
  "          max_iterations: 1                                            \n"
  "          convergence_tolerance: 1e-8                                  \n"
  "                                                                       \n"
  "      - ShearStressTransport:                                          \n"
  "          name: mySST                                                  \n"
  "          max_iterations: 1                                            \n"
  "          convergence_tolerance: 1e-8                                  \n"
  "                                                                       \n"
  "  time_step_control:                                                   \n"
  "    target_courant: 2.0                                                \n"
  "    time_step_change_factor: 1.2                                       \n"
  "                                                                       \n"
  "  solution_options:                                                    \n"
  "    name: myOptions                                                    \n"
  "    turbulence_model: sst_ams                                          \n"
  "    projected_timescale_type: momentum_diag_inv                        \n"
  "                                                                       \n"
  "    options:                                                           \n"
  "      - hybrid_factor:                                                 \n"
  "          turbulent_ke: 1.0                                            \n"
  "          specific_dissipation_rate: 1.0                               \n"
  "                                                                       \n"
  "      - alpha_upw:                                                     \n"
  "          velocity: 1.0                                                \n"
  "          turbulent_ke: 1.0                                            \n"
  "          specific_dissipation_rate: 1.0                               \n"
  "                                                                       \n"
  "      - upw_factor:                                                    \n"
  "          velocity: 1.0                                                \n"
  "          turbulent_ke: 0.0                                            \n"
  "          specific_dissipation_rate: 0.0                               \n";

// clang-format off
namespace hex8_golds {
namespace sst_ams_diff {
static constexpr double rhs[24] = {0.088069343129783, -0.054753698167151, 0, 0.043902291987831, 0.051585034673576, 0, -0.084720224280197, -0.058399243076388, 0, -0.047251410837417, 0.061567906569962, 0, 0.074491904516922, -0.054753698167151, 0, 0.0389564943094, 0.051534955236037, 0, -0.070999855021463, -0.049039131690076, 0, -0.042448543804859, 0.05225787462119, 0, };

static constexpr double lhs[24][24] ={
{0.37593796365399, 0, 0, -0.1654043452011, 0, 0, -0.12783144585235, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0.42106723690579, 0, 0, -0.082702172600548, 0, 0, -0.25566289170469, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, 0.37593796365399, 0, 0, -0.082702172600548, 0, 0, -0.12783144585235, 0, 0, 0, 0, 0, -0.1654043452011, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{-0.1654043452011, 0, 0, 0.35781821144931, 0, 0, 0, 0, 0, -0.10971169364766, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, -0.082702172600548, 0, 0, 0.38482773249642, 0, 0, 0, 0, 0, -0.21942338729532, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, -0.082702172600548, 0, 0, 0.35781821144931, 0, 0, 0, 0, 0, -0.10971169364766, 0, 0, 0, 0, 0, -0.1654043452011, 0, 0, 0, 0, 0, 0, },
{-0.12783144585235, 0, 0, 0, 0, 0, 0.55780460414762, 0, 0, -0.28647532993836, 0, 0, 0, 0, 0, 0, 0, 0, -0.14349782835692, 0, 0, 0, 0, 0, },
{0, -0.25566289170469, 0, 0, 0, 0, 0, 0.54239838503079, 0, 0, -0.14323766496918, 0, 0, 0, 0, 0, 0, 0, 0, -0.14349782835692, 0, 0, 0, 0, },
{0, 0, -0.12783144585235, 0, 0, 0, 0, 0, 0.55806476753537, 0, 0, -0.14323766496918, 0, 0, 0, 0, 0, 0, 0, 0, -0.28699565671384, 0, 0, 0, },
{0, 0, 0, -0.10971169364766, 0, 0, -0.28647532993836, 0, 0, 0.51835059583918, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12216357225317, 0, 0, },
{0, 0, 0, 0, -0.21942338729532, 0, 0, -0.14323766496918, 0, 0, 0.48482462451767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12216357225317, 0, },
{0, 0, 0, 0, 0, -0.10971169364766, 0, 0, -0.14323766496918, 0, 0, 0.49727650312317, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.24432714450633, },
{-0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.35806435685833, 0, 0, -0.1654043452011, 0, 0, -0.10995783905669, 0, 0, 0, 0, 0, },
{0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.38532002331447, 0, 0, -0.082702172600548, 0, 0, -0.21991567811338, 0, 0, 0, 0, },
{0, 0, -0.1654043452011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.35806435685833, 0, 0, -0.082702172600548, 0, 0, -0.10995783905669, 0, 0, 0, },
{0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, -0.1654043452011, 0, 0, 0.34672882483311, 0, 0, 0, 0, 0, -0.09862230703147, 0, 0, },
{0, 0, 0, 0, -0.082702172600548, 0, 0, 0, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0.36264895926404, 0, 0, 0, 0, 0, -0.19724461406294, 0, },
{0, 0, 0, 0, 0, -0.1654043452011, 0, 0, 0, 0, 0, 0, 0, 0, -0.082702172600548, 0, 0, 0.34672882483311, 0, 0, 0, 0, 0, -0.09862230703147, },
{0, 0, 0, 0, 0, 0, -0.14349782835692, 0, 0, 0, 0, 0, -0.10995783905669, 0, 0, 0, 0, 0, 0.49825733736442, 0, 0, -0.24480166995081, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, -0.14349782835692, 0, 0, 0, 0, 0, -0.21991567811338, 0, 0, 0, 0, 0, 0.4858143414457, 0, 0, -0.12240083497541, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, -0.28699565671384, 0, 0, 0, 0, 0, -0.10995783905669, 0, 0, 0, 0, 0, 0.51935433074594, 0, 0, -0.12240083497541, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12216357225317, 0, 0, 0, 0, 0, -0.09862230703147, 0, 0, -0.24480166995081, 0, 0, 0.46558754923545, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12216357225317, 0, 0, 0, 0, 0, -0.19724461406294, 0, 0, -0.12240083497541, 0, 0, 0.44180902129151, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.24432714450633, 0, 0, 0, 0, 0, -0.09862230703147, 0, 0, -0.12240083497541, 0, 0, 0.46535028651321, },
};

} // namespace sst_ams_diff
namespace ke_ams_diff {
static constexpr double rhs[24] = {0.085449014298766, -0.051898932126345, 0, 0.042307327929974, 0.04649316688805, 0, -0.080043249060471, -0.056932936468986, 0, -0.047713093168269, 0.062338701707281, 0, 0.071637798107362, -0.051898932126345, 0, 0.037277437925548, 0.046325139282165, 0, -0.065799428391419, -0.047366299227838, 0, -0.043115807641491, 0.052940092072017, 0, };

static constexpr double lhs[24][24] ={
{0.38976292287141, 0, 0, -0.17246171497369, 0, 0, -0.13107035041089, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0.43460241579546, 0, 0, -0.086230857486843, 0, 0, -0.26214070082177, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, 0.38976292287141, 0, 0, -0.086230857486843, 0, 0, -0.13107035041089, 0, 0, 0, 0, 0, -0.17246171497369, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{-0.17246171497369, 0, 0, 0.3717583550199, 0, 0, 0, 0, 0, -0.11306578255937, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, -0.086230857486843, 0, 0, 0.39859328009243, 0, 0, 0, 0, 0, -0.22613156511874, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, -0.086230857486843, 0, 0, 0.3717583550199, 0, 0, 0, 0, 0, -0.11306578255937, 0, 0, 0, 0, 0, -0.17246171497369, 0, 0, 0, 0, 0, 0, },
{-0.13107035041089, 0, 0, 0, 0, 0, 0.57045386739397, 0, 0, -0.29264239396569, 0, 0, 0, 0, 0, 0, 0, 0, -0.1467411230174, 0, 0, 0, 0, 0, },
{0, -0.26214070082177, 0, 0, 0, 0, 0, 0.55520302082201, 0, 0, -0.14632119698285, 0, 0, 0, 0, 0, 0, 0, 0, -0.1467411230174, 0, 0, 0, 0, },
{0, 0, -0.13107035041089, 0, 0, 0, 0, 0, 0.57087379342852, 0, 0, -0.14632119698285, 0, 0, 0, 0, 0, 0, 0, 0, -0.29348224603479, 0, 0, 0, },
{0, 0, 0, -0.11306578255937, 0, 0, -0.29264239396569, 0, 0, 0.5312301714117, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12552199488664, 0, 0, },
{0, 0, 0, 0, -0.22613156511874, 0, 0, -0.14632119698285, 0, 0, 0.49797475698823, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12552199488664, 0, },
{0, 0, 0, 0, 0, -0.11306578255937, 0, 0, -0.14632119698285, 0, 0, 0.51043096931549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.25104398977328, },
{-0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37217828105445, 0, 0, -0.17246171497369, 0, 0, -0.11348570859392, 0, 0, 0, 0, 0, },
{0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.39943313216153, 0, 0, -0.086230857486843, 0, 0, -0.22697141718784, 0, 0, 0, 0, },
{0, 0, -0.17246171497369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37217828105445, 0, 0, -0.086230857486843, 0, 0, -0.11348570859392, 0, 0, 0, },
{0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, -0.17246171497369, 0, 0, 0.36084580821197, 0, 0, 0, 0, 0, -0.10215323575144, 0, 0, },
{0, 0, 0, 0, -0.086230857486843, 0, 0, 0, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0.37676818647657, 0, 0, 0, 0, 0, -0.20430647150289, 0, },
{0, 0, 0, 0, 0, -0.17246171497369, 0, 0, 0, 0, 0, 0, 0, 0, -0.086230857486843, 0, 0, 0.36084580821197, 0, 0, 0, 0, 0, -0.10215323575144, },
{0, 0, 0, 0, 0, 0, -0.1467411230174, 0, 0, 0, 0, 0, -0.11348570859392, 0, 0, 0, 0, 0, 0.51206418212822, 0, 0, -0.2518373505169, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, -0.1467411230174, 0, 0, 0, 0, 0, -0.22697141718784, 0, 0, 0, 0, 0, 0.49963121546369, 0, 0, -0.12591867525845, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, -0.29348224603479, 0, 0, 0, 0, 0, -0.11348570859392, 0, 0, 0, 0, 0, 0.53288662988716, 0, 0, -0.12591867525845, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12552199488664, 0, 0, 0, 0, 0, -0.10215323575144, 0, 0, -0.2518373505169, 0, 0, 0.47951258115498, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.12552199488664, 0, 0, 0, 0, 0, -0.20430647150289, 0, 0, -0.12591867525845, 0, 0, 0.45574714164797, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.25104398977328, 0, 0, 0, 0, 0, -0.10215323575144, 0, 0, -0.12591867525845, 0, 0, 0.47911590078317, },
};

} // namespace ke_ams_diff
} // namespace hex8_golds
// clang-format on
} // anonymous namespace

TEST_F(AMSKernelHex8Mesh, ngp_sst_ams_diff)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmSSTAMSSettings);

  realm_node[0]["solution_options"]["turbulence_model"] = "sst_ams";

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = false;
  solnOpts_.alphaMap_["velocity"] = 0.0;
  solnOpts_.alphaUpwMap_["velocity"] = 0.0;
  solnOpts_.upwMap_["velocity"] = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::AMSEdgeKernelHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0],
    unit_test_utils::get_default_inputs(), realm_node[0]);

  helperObjs.edgeAlg->add_kernel<sierra::nalu::MomentumSSTAMSDiffEdgeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::sst_ams_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
}

TEST_F(AMSKernelHex8Mesh, ngp_sstlr_ams_diff)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmSSTAMSSettings);

  realm_node[0]["solution_options"]["turbulence_model"] = "sstlr_ams";

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = false;
  solnOpts_.alphaMap_["velocity"] = 0.0;
  solnOpts_.alphaUpwMap_["velocity"] = 0.0;
  solnOpts_.upwMap_["velocity"] = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::AMSEdgeKernelHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0],
    unit_test_utils::get_default_inputs(), realm_node[0]);

  helperObjs.edgeAlg->add_kernel<sierra::nalu::MomentumSSTLRAMSDiffEdgeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::sst_ams_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
}

TEST_F(AMSKernelHex8Mesh, ngp_ko_ams_diff)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmSSTAMSSettings);

  realm_node[0]["solution_options"]["turbulence_model"] = "ko_ams";

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = false;
  solnOpts_.alphaMap_["velocity"] = 0.0;
  solnOpts_.alphaUpwMap_["velocity"] = 0.0;
  solnOpts_.upwMap_["velocity"] = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::AMSEdgeKernelHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0],
    unit_test_utils::get_default_inputs(), realm_node[0]);

  helperObjs.edgeAlg->add_kernel<sierra::nalu::MomentumKOAMSDiffEdgeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::sst_ams_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
}

TEST_F(AMSKernelHex8Mesh, ngp_ke_ams_diff)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmSSTAMSSettings);

  realm_node[0]["solution_options"]["turbulence_model"] = "ke_ams";

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = false;
  solnOpts_.alphaMap_["velocity"] = 0.0;
  solnOpts_.alphaUpwMap_["velocity"] = 0.0;
  solnOpts_.upwMap_["velocity"] = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::AMSEdgeKernelHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0],
    unit_test_utils::get_default_inputs(), realm_node[0]);

  helperObjs.edgeAlg->add_kernel<sierra::nalu::MomentumKEAMSDiffEdgeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::ke_ams_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
}
