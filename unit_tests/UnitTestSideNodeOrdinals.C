#include <gtest/gtest.h>
#include <stk_topology/topology.hpp>
#include "UnitTestUtils.h"

#include <master_element/MasterElementFactory.h>
namespace {

  void side_node_ordinals_are_same_as_stk(stk::topology topo)
  {
    auto* me = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

    for (unsigned side_ordinal = 0; side_ordinal < topo.num_sides(); ++side_ordinal) {
      int numSideNodes = topo.side_topology(side_ordinal).num_nodes();
      std::vector<unsigned> topoSideNodeOrdinals(numSideNodes);
      topo.side_node_ordinals(side_ordinal, topoSideNodeOrdinals.begin());
      const int* meSideNodeOrdinals = me->side_node_ordinals(side_ordinal);

      for (int n = 0; n < numSideNodes; ++n) {
        EXPECT_EQ(static_cast<int>(topoSideNodeOrdinals[n]), meSideNodeOrdinals[n]);
      }
    }
  }
}

#define TEST_ALL_TOPOS(x, y) \
    TEST(x, tri##_##y)   { y(stk::topology::TRI_3_2D); } \
    TEST(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST(x, quad9##_##y)  { y(stk::topology::QUAD_9_2D); } \
    TEST(x, tet##_##y)   { y(stk::topology::TET_4); } \
    TEST(x, pyr##_##y)   { y(stk::topology::PYRAMID_5); } \
    TEST(x, wedge##_##y) { y(stk::topology::WEDGE_6); } \
    TEST(x, hex8##_##y)   { y(stk::topology::HEX_8); } \
    TEST(x, hex27##_##y)   { y(stk::topology::HEX_27); }

TEST_ALL_TOPOS(MasterElementSideNodeOrdinals, side_node_ordinals_are_same_as_stk)



