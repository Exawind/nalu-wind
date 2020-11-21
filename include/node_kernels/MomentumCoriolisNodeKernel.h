// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMCORIOLISSRCNODEKERNEL_H
#define MOMENTUMCORIOLISSRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "CoriolisSrc.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumCoriolisNodeKernel : public NGPNodeKernel<MomentumCoriolisNodeKernel>
{
public:
  MomentumCoriolisNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&);

  MomentumCoriolisNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumCoriolisNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  const CoriolisSrc cor_;

  stk::mesh::NgpField<double> dualNodalVolume_;
  stk::mesh::NgpField<double> densityNp1_;
  stk::mesh::NgpField<double> velocityNp1_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1ID_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMCORIOLISSRCNODEKERNEL_H */
