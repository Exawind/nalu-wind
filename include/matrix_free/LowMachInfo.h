// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOWMACH_INFO_H
#define LOWMACH_INFO_H

#include "matrix_free/LinSysInfo.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

enum class GradTurbModel { LAM, WALE, SMAG };

struct lowmach_info
{
  static constexpr int num_physics_fields = 10;
  static constexpr int num_coefficient_fields = 3;
  static constexpr auto coord_name = "coordinates";
  static constexpr auto density_name = "density";
  static constexpr auto velocity_name = "velocity";
  static constexpr auto velocity_bc_name = "velocity_bc";
  static constexpr auto pressure_name = "pressure";
  static constexpr auto pressure_grad_name = "dpdx";
  static constexpr auto viscosity_name = "viscosity";
  static constexpr auto scaled_filter_length_name = "scaled_filter_length";
  static constexpr auto force_name = "body_force";
  static constexpr auto gid_name = linsys_info::gid_name;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
