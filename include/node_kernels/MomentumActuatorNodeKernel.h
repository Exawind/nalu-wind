// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMACTUATORNODEKERNEL_H
#define MOMENTUMACTUATORNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MomentumActuatorNodeKernel
  : public NGPNodeKernel<MomentumActuatorNodeKernel>
{
public:
  MomentumActuatorNodeKernel(const stk::mesh::MetaData&);

  MomentumActuatorNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumActuatorNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> actuatorSrc_;
  stk::mesh::NgpField<double> actuatorSrcLHS_;
  const int nDim_;

  const unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  const unsigned actuatorSrcID_{stk::mesh::InvalidOrdinal};
  const unsigned actuatorSrcLHSID_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif
