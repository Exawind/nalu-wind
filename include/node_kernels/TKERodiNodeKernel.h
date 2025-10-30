// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TKERODINODEKERNEL_H
#define TKERODINODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;
class SolutionOptions;

class TKERodiNodeKernel : public NGPNodeKernel<TKERodiNodeKernel>
{
public:
  TKERodiNodeKernel(const stk::mesh::MetaData&, const SolutionOptions&);

  TKERodiNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~TKERodiNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> dhdx_;
  stk::mesh::NgpField<double> specificHeat_;
  stk::mesh::NgpField<double> tvisc_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  const unsigned dhdxID_{stk::mesh::InvalidOrdinal};
  const unsigned specificHeatID_{stk::mesh::InvalidOrdinal};
  const unsigned tviscID_{stk::mesh::InvalidOrdinal};
  const unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType gravity_[NodeKernelTraits::NDimMax];
  NodeKernelTraits::DblType turbPr_;
  const NodeKernelTraits::DblType beta_;
  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif
