/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
