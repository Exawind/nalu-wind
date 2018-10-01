/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/NodeMapMaker.h>
#include <element_promotion/ElementDescription.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>

namespace sierra {
namespace nalu{

Kokkos::View<int*> make_node_map(int p, int dim, bool isPromoted)
{
  // generally, we don't need the full element description, but instead
  // we just need to know how to permute the node ordering to get
  // into and tensor product form

  if (!isPromoted && dim == 3) {
    ThrowRequireMsg(p == 2, "Only hex 27");
    Kokkos::View<int*> node_map("node_map", (p+1)*(p+1)*(p+1));

    std::array<int,27> stkNodeMap = {{
        0,  8,  1, // bottom front edge
        11, 21,  9, // bottom mid-front edge
        3, 10,  2, // bottom back edge
        12, 25, 13, // mid-top front edge
        23, 20, 24, // mid-top mid-front edge
        15, 26, 14, // mid-top back edge
        4, 16,  5, // top front edge
        19, 22, 17, // top mid-front edge
        7, 18,  6  // top back edge
    }};

    for (int k = 0; k < 27; ++k) {
      node_map[k] = stkNodeMap[k];
    }
    return node_map;
  }
  ThrowRequire(dim != 2 || isPromoted);

  if (dim == 3) {
    Kokkos::View<int*> node_map("node_map", (p+1)*(p+1)*(p+1));
    int nodes1D = p+1;
    auto desc = ElementDescription::create(dim, p);
    for (int k = 0; k < nodes1D; ++k) {
      for (int j = 0; j < nodes1D; ++j) {
        for (int i = 0; i < nodes1D; ++i) {
          node_map(k*(nodes1D)*(nodes1D) + j*(nodes1D) + i) = desc->node_map(i,j,k);
        }
      }
    }
    return node_map;
  }
  else {
    Kokkos::View<int*> node_map("node_map", (p+1)*(p+1));
    int nodes1D = p + 1;
    auto desc = ElementDescription::create(dim, p);
      for (int j = 0; j < nodes1D; ++j) {
        for (int i = 0; i < nodes1D; ++i) {
          node_map(j * nodes1D + i) = desc->node_map(i,j);
        }

    }
    return node_map;
  }
}

Kokkos::View<int*> make_inverse_node_map(int p, int dim, bool isPromoted)
{
  auto node_map = make_node_map(p, dim, isPromoted);
  Kokkos::View<int*> inverseNodeMap("inverse_node_map", node_map.size());
  for (unsigned k = 0; k < node_map.size(); ++k) {
    inverseNodeMap[node_map[k]] = k;
  }
  return inverseNodeMap;
}


Kokkos::View<int*> make_node_map(int p, stk::topology baseTopo, bool isPromoted)
    { return make_node_map(p, baseTopo.dimension(), isPromoted); }

Kokkos::View<int*> make_inverse_node_map(int p, stk::topology baseTopo, bool isPromoted)
    { return make_inverse_node_map(p, baseTopo.dimension(), isPromoted); }

} // namespace naluUnit
} // namespace Sierra


