// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMSYMMETRYEDGEKERNEL_H
#define MOMENTUMSYMMETRYEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;
class ElemDataRequests;
class MasterElement;

template<typename BcAlgTraits>
class MomentumSymmetryEdgeKernel: public NGPKernel<MomentumSymmetryEdgeKernel<BcAlgTraits>>
{
public:
  MomentumSymmetryEdgeKernel(
    const stk::mesh::MetaData&,
    const SolutionOptions&,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&,
    ElemDataRequests&);

  KOKKOS_FORCEINLINE_FUNCTION
  MomentumSymmetryEdgeKernel() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  virtual ~MomentumSymmetryEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:
  unsigned coordinates_    {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_    {stk::mesh::InvalidOrdinal};
  unsigned viscosity_      {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned dudx_           {stk::mesh::InvalidOrdinal};

  const DoubleType includeDivU_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMSYMMETRYEDGEKERNEL_H */
