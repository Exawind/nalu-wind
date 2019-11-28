// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef WALLDISTNODEKERNEL_H
#define WALLDISTNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class WallDistNodeKernel : public NGPNodeKernel<WallDistNodeKernel>
{
public:
  WallDistNodeKernel(stk::mesh::BulkData&);

  KOKKOS_FUNCTION WallDistNodeKernel() = default;

  KOKKOS_FUNCTION ~WallDistNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* WALLDISTNODEKERNEL_H */
