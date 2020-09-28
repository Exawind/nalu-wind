// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TransportCoefficients_H
#define TransportCoefficients_H

#include "matrix_free/LowMachFields.h"
#include "matrix_free/KokkosViewTypes.h"

#include "matrix_free/LowMachInfo.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p>
struct transport_coefficients_t
{
  static void invoke(
    GradTurbModel model,
    const const_elem_mesh_index_view<p>& conn,
    const stk::mesh::NgpField<double>& rho_f,
    const stk::mesh::NgpField<double>& mu_f,
    const_scalar_view<p> filter_scale,
    const_vector_view<p> xc,
    const_vector_view<p> vel,
    const_scalar_view<p> unscaled_vol,
    const_scs_vector_view<p> unscaled_diff,
    scalar_view<p> rho,
    scalar_view<p> visc,
    scalar_view<p> vol,
    scs_vector_view<p> diff);
};
} // namespace impl
P_INVOKEABLE(transport_coefficients)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
