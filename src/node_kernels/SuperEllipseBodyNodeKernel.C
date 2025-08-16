// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "Realm.h"
#include "SolutionOptions.h"
#include "node_kernels/SuperEllipseBodyNodeKernel.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/Types.hpp"
#include <aero/aero_utils/WienerMilenkovic.h>

namespace sierra {
namespace nalu {

SuperEllipseBodyNodeKernel::SuperEllipseBodyNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts,
  const SuperEllipseBodySrc& seb)
  : NGPNodeKernel<SuperEllipseBodyNodeKernel>(),
    seb_(seb),
    seb_loc_(seb.get_loc()),
    seb_orient_(seb.get_orient()),
    seb_dim_(seb.get_dim())
{
  const auto& meta = bulk.mesh_meta_data();

  velocityNp1ID_ = get_field_ordinal(meta, "velocity");
  densityNp1ID_ = get_field_ordinal(meta, "density");
  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
}

void
SuperEllipseBodyNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  velocityNp1_ = fieldMgr.get_field<double>(velocityNp1ID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
  coordsNp1_ = fieldMgr.get_field<double>(coordinatesID_);

  seb_loc_ = seb_.get_loc();
  seb_orient_ = seb_.get_orient();
  seb_dim_ = seb_.get_dim();
  dt_ = realm.get_time_step();
}

KOKKOS_FUNCTION
void
SuperEllipseBodyNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  NodeKernelTraits::DblType dualVol = dualNodalVolume_.get(node, 0);

  const double fac = rhoNp1 * dualVol / dt_;

  vs::Vector coords;
  for (int i = 0; i < NodeKernelTraits::NDimMax; ++i)
    coords[i] = coordsNp1_.get(node, i);

  vs::Vector coords_t = wmp::rotate(seb_orient_, coords - seb_loc_);

  if ( ( stk::math::pow(coords_t[0]/seb_dim_[0],6)
  + stk::math::pow(coords_t[1]/seb_dim_[1],6)
  +  stk::math::pow(coords_t[2]/seb_dim_[2],6) - 1.0 ) < 0.0 ) {
    for (int i = 0; i < NodeKernelTraits::NDimMax; ++i) {
      rhs(i) += fac * velocityNp1_.get(node, i);
      lhs(i,i) -= fac;
    }
  }

}

} // namespace nalu
} // namespace sierra
