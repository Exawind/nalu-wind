// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOCAL_DUAL_NODAL_VOLUME_H
#define LOCAL_DUAL_NODAL_VOLUME_H

#include "matrix_free/PolynomialOrders.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"

#include "Tpetra_CrsMatrix.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Selector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p>
struct local_dual_nodal_volume_t
{
  static void invoke(
    const stk::mesh::NgpMesh& mesh,
    const stk::mesh::Selector& sel,
    const stk::mesh::NgpField<double>& coords,
    stk::mesh::NgpField<double>& dnv);
};
} // namespace impl
SWITCH_INVOKEABLE(local_dual_nodal_volume)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
