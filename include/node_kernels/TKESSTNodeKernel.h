// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TKESSTNODEKERNEL_H
#define TKESSTNODEKERNEL_H

#include "LinearSystem.h"
#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"
#include <FieldManager.h>

namespace sierra {
namespace nalu {

class Realm;

class TKESSTNodeKernel : public NGPNodeKernel<TKESSTNodeKernel>
{
public:
  TKESSTNodeKernel(
    const stk::mesh::MetaData&, const FieldManager&, stk::mesh::Part* part);

  TKESSTNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~TKESSTNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> tke_;
  stk::mesh::NgpField<double> sdr_;
  stk::mesh::NgpField<double> density_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> dudx_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  NodeKernelTraits::DblType tkeAmb_;
  NodeKernelTraits::DblType sdrAmb_;

  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* TKESSTNODEKERNEL_H */
