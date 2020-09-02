#include <gtest/gtest.h>
#include <limits>

#include <memory>
#include <vector>

#include "element_promotion/HexNElementDescription.h"
#include "UnitTestUtils.h"

namespace {


TEST(ElementDescriptionTestHex, node_map_P2)
{
  // NOTE: the center node has been moved to being numbered last
  // this is just to avoid a re-mapping when doing static condensation
  std::vector<int> stkNodeMap =
  {
      0, 8, 1,
      11, 20, 9,
      3, 10, 2,
      12, 24, 13,
      22, 26, 23,
      15, 25, 14,
      4, 16, 5,
      19, 21, 17,
      7, 18, 6
  };

  auto elem = sierra::nalu::HexNElementDescription(2);
  for (unsigned j = 0; j < stkNodeMap.size(); ++j) {
    EXPECT_EQ(stkNodeMap[j], elem.node_map(j));
  }
}

TEST(ElementDescriptionTestHex, node_map_P3)
{
  std::vector<int> intendedNodeMap =
  {
      0, 8, 9, 1,
      15, 32, 34, 10,
      14, 33, 35, 11,
      3, 13, 12, 2,
      16, 48, 49, 18,
      40, 56, 57, 44,
      42, 58, 59, 45,
      22, 53, 52, 20,
      17, 50, 51, 19,
      41, 60, 61, 46,
      43, 62, 63, 47,
      23, 55, 54, 21,
      4, 24, 25, 5,
      31, 36, 37, 26,
      30, 38, 39, 27,
      7, 29, 28, 6
  };

  auto elem = sierra::nalu::HexNElementDescription(3);
  for (unsigned j = 0; j < intendedNodeMap.size(); ++j) {
    EXPECT_EQ(intendedNodeMap[j], elem.node_map(j));
  }
}

TEST(ElementDescriptionTestHex, side_ordinal_map_P2)
{
  std::vector<std::vector<int>> sideNodeMap =
  {
    {0, 1, 5, 4, 8, 13, 16, 12, 24},
    {1, 2, 6, 5, 9, 14, 17, 13, 23},
    {2, 3, 7, 6, 10, 15, 18, 14, 25},
    {0, 4, 7, 3, 12, 19, 15, 11, 22},
    {0, 3, 2, 1, 11, 10, 9, 8, 20},
    {4, 5, 6, 7, 16, 17, 18, 19, 21}
  };

  auto elem = sierra::nalu::HexNElementDescription(2);
  for (int side_ordinal = 0; side_ordinal < elem.numFaces; ++side_ordinal) {
    auto side_node_ordinals = elem.side_node_ordinals(side_ordinal);
    auto expected_side_node_ordinals = sideNodeMap.at(side_ordinal);
    for (unsigned n = 0; n < expected_side_node_ordinals.size(); ++n) {
      EXPECT_EQ(side_node_ordinals[n], expected_side_node_ordinals.at(n));
    }
  }
}

TEST(ElementDescriptionTestHex, side_ordinal_map_P3)
{
  std::vector<std::vector<int>> sideNodeMap =
  {
      { 0, 1, 5, 4, 8, 9, 18, 19, 25, 24, 17, 16, 48, 49, 50, 51 },
      { 1, 2, 6, 5, 10, 11, 20, 21, 27, 26, 19, 18, 44, 45, 46, 47 },
      { 2, 3, 7, 6, 12, 13, 22, 23, 29, 28, 21, 20, 52, 53, 54, 55 },
      { 0, 4, 7, 3, 16, 17, 31, 30, 23, 22, 14, 15, 40, 41, 42, 43 },
      { 0, 3, 2, 1, 15, 14, 13, 12, 11, 10, 9, 8, 32, 33, 34, 35 },
      { 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37, 38, 39 }
  };

  auto elem = sierra::nalu::HexNElementDescription(3);
  for (int side_ordinal = 0; side_ordinal < elem.numFaces; ++side_ordinal) {
    auto side_node_ordinals = elem.side_node_ordinals(side_ordinal);
    auto expected_side_node_ordinals = sideNodeMap.at(side_ordinal);
    for (unsigned n = 0; n < expected_side_node_ordinals.size(); ++n) {
      EXPECT_EQ(side_node_ordinals[n], expected_side_node_ordinals.at(n));
    }
  }
}

}

