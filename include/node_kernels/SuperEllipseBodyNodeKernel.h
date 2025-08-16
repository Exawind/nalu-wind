// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SUPERELLIPSEBODYSRCNODEKERNEL_H
#define SUPERELLIPSEBODYSRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "SuperEllipseBodySrc.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

#include "vs/vector_space.h"

namespace sierra {
namespace nalu {

class SolutionOptions;

class SuperEllipseBodyNodeKernel
  : public NGPNodeKernel<SuperEllipseBodyNodeKernel>
{
public:
  SuperEllipseBodyNodeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&, 
    const SuperEllipseBodySrc& seb);

  SuperEllipseBodyNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~SuperEllipseBodyNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:

  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> densityNp1_;
  stk::mesh::NgpField<double> velocityNp1_;
  stk::mesh::NgpField<double> coordsNp1_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_{stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};

  // Reference to the SuperEllipseBodySrc object
  const SuperEllipseBodySrc& seb_;

  // Location of the Super Ellipse center
  vs::Vector seb_loc_; 
  // Orientation of the Super Ellipse
  vs::Vector seb_orient_; 
  // Dimensions of the Super Ellipse
  vs::Vector seb_dim_; 

  // Time step
  double dt_;
};

} // namespace nalu
} // namespace sierra

#endif /* SUPERELLIPSEBODYSRCNODEKERNEL_H */
