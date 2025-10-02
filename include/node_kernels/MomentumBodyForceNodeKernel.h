// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMBODYFORCENODEKERNEL_H
#define MOMENTUMBODYFORCENODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MomentumBodyForceNodeKernel
  : public NGPNodeKernel<MomentumBodyForceNodeKernel>
{
public:
  MomentumBodyForceNodeKernel(
    const stk::mesh::BulkData&, const std::vector<double>&);

  MomentumBodyForceNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumBodyForceNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;

   NodeKernelTraits::DblType
    forceVector_[NodeKernelTraits::NDimMax];

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};

  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMBODYFORCENODEKERNEL_H */
