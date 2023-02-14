#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "UnitTestUtils.h"

#include "KokkosInterface.h"
#include "aero/fsi/FSIturbine.h"

namespace {

const std::string fsiInputs =
"tower_parts: [block_1] \n"
"hub_parts: [block_2]\n"
"nacelle_parts: [block_3,block_4] \n"
"blade_parts:\n"
"  - [block_3]\n"
"  - [block_4]\n"
"deflection_ramping:\n"
"  span_ramp_distance: 10.0\n"
"  zero_theta_ramp_angle: 180.0\n"
"  theta_ramp_span: 15.0\n"
"  temporal_ramp_start: 0\n"
"  temporal_ramp_end: 10\n"
"tower_boundary_parts: [block_1] \n"
"hub_boundary_parts: [block_2]\n"
"nacelle_boundary_parts: [block_3,block_4] \n"
"blade_boundary_parts:\n"
"  - [block_3]\n"
"  - [block_4]\n"
;

YAML::Node create_fsi_yaml_node()
{
  YAML::Node yamlNode = YAML::Load(fsiInputs);
  return yamlNode;
}

TEST_F(CylinderMesh, construct_FSIturbine)
{
  const double innerRadius = 1.0;
  const double outerRadius = 2.0;
  fill_mesh_and_initialize_test_fields(20, 20, 20, innerRadius, outerRadius);

  YAML::Node yamlNode = create_fsi_yaml_node();
  EXPECT_NO_THROW(sierra::nalu::fsiTurbine(0, yamlNode));
}

} // namespace

