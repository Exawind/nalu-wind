/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/ElementDescription.h>

#include <element_promotion/QuadNElementDescription.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadratureRule.h>
#include <nalu_make_unique.h>

#include <cmath>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {

int poly_order_from_topology(int dimension, stk::topology topo)
{
  /**
   * "super topologies" in STK are essentially just the number of nodes in the element.
   * Even restricting ourselves to quads/hexs, we need slightly more
   * information to reconstruct the ElementDescription---namely the dimension.
   *
   * Alternatively, we could restrict the valid polynomial orders to be < 7,
   * and then we could base everything on topology alone
   */

  if (topo == stk::topology::HEX_27 || topo == stk::topology::QUADRILATERAL_9_2D) {
    return 2;
  }
  else if (!topo.is_super_topology()) {
    return 1;
  }

  if (topo.is_superedge()) {
    return (topo.num_nodes()-1);
  }

  if (topo.is_superface()) {
    int nodes1D = std::sqrt(topo.num_nodes()+1);
    return (nodes1D-1);
  }

  const int nodes1D = (dimension == 2) ? std::sqrt(topo.num_nodes()+1) : std::cbrt(topo.num_nodes()+1);

  const int polynomial_order = nodes1D-1;
  return polynomial_order;
}

std::unique_ptr<ElementDescription>
ElementDescription::create(int dimension, stk::topology superTopo)
{
  ThrowRequireMsg(superTopo.is_super_topology(), "Only valid for super topologies");
  return ElementDescription::create(dimension, poly_order_from_topology(dimension, superTopo));
}

//--------------------------------------------------------------------------
std::unique_ptr<ElementDescription>
ElementDescription::create(int dimension, int order)
{
  auto nodeLocations1D = gauss_lobatto_legendre_rule(order + 1).first;
  if (dimension == 2) {
    return make_unique<QuadNElementDescription>(nodeLocations1D);
  }
  return make_unique<HexNElementDescription>(nodeLocations1D);
}
//--------------------------------------------------------------------------
ElementDescription::ElementDescription()  = default;
//--------------------------------------------------------------------------
ElementDescription::~ElementDescription() = default;
//--------------------------------------------------------------------------

} // namespace nalu
}  // namespace sierra
