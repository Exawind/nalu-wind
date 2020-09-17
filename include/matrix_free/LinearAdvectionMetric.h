// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LINEAR_ADVECTION_METRIC_H
#define LINEAR_ADVECTION_METRIC_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {

namespace impl {

template <int p>
struct linear_advection_metric_t
{
  static void invoke(
    double scaling,
    const_scs_vector_view<p> areas,
    const_scs_vector_view<p> laplacian_metric,
    scalar_view<p> density,
    vector_view<p> velocity,
    vector_view<p> proj_pressure_gradient,
    scalar_view<p> pressure,
    scs_scalar_view<p>& mdot);
};

} // namespace impl
P_INVOKEABLE(linear_advection_metric)

} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
