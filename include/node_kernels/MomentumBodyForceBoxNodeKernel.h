// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMBODYFORCEBOXNODEKERNEL_H
#define MOMENTUMBODYFORCEBOXNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MomentumBodyForceBoxNodeKernel
  : public NGPNodeKernel<MomentumBodyForceBoxNodeKernel>
{
public:
  MomentumBodyForceBoxNodeKernel(
    const stk::mesh::BulkData&,
    const std::string,
    const std::vector<double>&,
    const std::vector<double>& = std::vector<double>());

  KOKKOS_FUNCTION
  MomentumBodyForceBoxNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumBodyForceBoxNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  NALU_ALIGNED NodeKernelTraits::DblType
    forceVector_[NodeKernelTraits::NDimMax];
  NALU_ALIGNED NodeKernelTraits::DblType lo_[NodeKernelTraits::NDimMax];
  NALU_ALIGNED NodeKernelTraits::DblType hi_[NodeKernelTraits::NDimMax];

  unsigned coordinatesID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};

  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMBODYFORCEBOXNODEKERNEL_H */
