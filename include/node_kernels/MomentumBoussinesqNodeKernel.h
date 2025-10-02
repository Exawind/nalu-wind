// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMBOUSSINESQNODEKERNEL_h
#define MOMENTUMBOUSSINESQNODEKERNEL_h

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumBoussinesqNodeKernel
  : public NGPNodeKernel<MomentumBoussinesqNodeKernel>
{
public:
  MomentumBoussinesqNodeKernel(
    const stk::mesh::BulkData&, const SolutionOptions&);

  MomentumBoussinesqNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumBoussinesqNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> temperature_;
  const int nDim_;
  NodeKernelTraits::DblType tRef_;
  NodeKernelTraits::DblType rhoRef_;
  NodeKernelTraits::DblType beta_;

  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned temperatureID_{stk::mesh::InvalidOrdinal};

   NodeKernelTraits::DblType gravity_[NodeKernelTraits::NDimMax];
};

} // namespace nalu
} // namespace sierra

#endif
