// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef AFTNTILDANODEKERNEL_H
#define AFTNTILDANODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "stk_mesh/base/NgpField.hpp"

namespace sierra {
namespace kynema_ugf {

class Realm;

class AFTNTildaNodeKernel : public NGPNodeKernel<AFTNTildaNodeKernel>
{
public:
  AFTNTildaNodeKernel(const stk::mesh::MetaData&);

  AFTNTildaNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~AFTNTildaNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> viscosity_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> minDistance_;
  stk::mesh::NgpField<double> dwalldistdx_;
  stk::mesh::NgpField<double> dnDotVdx_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  unsigned densityID_{stk::mesh::InvalidOrdinal};
  unsigned viscosityID_{stk::mesh::InvalidOrdinal};
  unsigned tviscID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned minDistanceID_{stk::mesh::InvalidOrdinal};
  unsigned dwalldistdxID_{stk::mesh::InvalidOrdinal};
  unsigned dnDotVdxID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};

  const int nDim_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif /* AFTNTILDANODEKERNEL_H */