// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMABLFORCENODEKERNEL_H
#define MOMENTUMABLFORCENODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "wind_energy/ABLSrcInterp.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumABLForceNodeKernel : public NGPNodeKernel<MomentumABLForceNodeKernel>
{
public:
  MomentumABLForceNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&);

  MomentumABLForceNodeKernel() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~MomentumABLForceNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  stk::mesh::NgpField<double> coordinates_;
  stk::mesh::NgpField<double> dualNodalVolume_;

  ABLVectorInterpolator ablSrc_;

  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};

  const int nDim_;
};

}  // nalu
}  // sierra


#endif /* MOMENTUMABLFORCENODEKERNEL_H */
