/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMBODYFORCENODEKERNEL_H
#define MOMENTUMBODYFORCENODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class MomentumBodyForceNodeKernel : public NGPNodeKernel<MomentumBodyForceNodeKernel>
{
public:
  MomentumBodyForceNodeKernel(
    const stk::mesh::BulkData&,
    const std::vector<double>&);

  KOKKOS_FUNCTION
  MomentumBodyForceNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumBodyForceNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  NALU_ALIGNED NodeKernelTraits::DblType forceVector_[NodeKernelTraits::NDimMax];

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};

  const int nDim_;
};

}  // nalu
}  // sierra



#endif /* MOMENTUMBODYFORCENODEKERNEL_H */
