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

Kokkos::View<int*> make_node_map(int p, bool isPromoted)
{
  // generally, we don't need the full element description, but instead
  // we just need to know how to permute the node ordering to get
  // into and tensor product form

  if (!isPromoted && p == 2) {
    // The default exodus standard for Hex27 has the center node
    // as #20, but the promoted elements have the interior nodes
    // at the end for easy condensation
    constexpr int npe = 27;
    Kokkos::View<int*> node_map("node_map", npe);
    std::array<int, npe> stkNodeMap = { {
        0, 8, 1,    // bottom front edge
        11, 21, 9,  // bottom mid-front edge
        3, 10, 2,   // bottom back edge
        12, 25, 13, // mid-top front edge
        23, 20, 24, // mid-top mid-front edge
        15, 26, 14, // mid-top back edge
        4, 16, 5,   // top front edge
        19, 22, 17, // top mid-front edge
        7, 18, 6    // top back edge
    } };

    for (int k = 0; k < npe; ++k) {
      node_map[k] = stkNodeMap[k];
    }
    return node_map;
  }
  Kokkos::View<int*> node_map("node_map", (p+1)*(p+1)*(p+1));
  int nodes1D = p+1;
  auto desc = ElementDescription::create(3, p);
  for (int k = 0; k < nodes1D; ++k) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int i = 0; i < nodes1D; ++i) {
        node_map(k*(nodes1D)*(nodes1D) + j*(nodes1D) + i) = desc->node_map(i,j,k);
      }
    }
  }
  return node_map;
}

Kokkos::View<int*> make_inverse_node_map(int p, bool isPromoted)
{
  auto node_map = make_node_map(p, isPromoted);
  Kokkos::View<int*> inverseNodeMap("inverse_node_map", node_map.size());
  for (unsigned k = 0; k < node_map.size(); ++k) {
    inverseNodeMap[node_map[k]] = k;
  }
  return inverseNodeMap;
}

Kokkos::View<int*> make_node_map(int p, stk::topology topo)
    { return make_node_map(p, topo.is_super_topology()); }

Kokkos::View<int*> make_inverse_node_map(int p, stk::topology topo)
    { return make_inverse_node_map(p, topo.is_super_topology()); }

} // namespace naluUnit
} // namespace Sierra


