// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SCALAREDGEOPENSOLVERALG_H
#define SCALAREDGEOPENSOLVERALG_H

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
class ScalarEdgeOpenSolverAlg: public NGPKernel<ScalarEdgeOpenSolverAlg<BcAlgTraits>>
{
public:
  ScalarEdgeOpenSolverAlg(
    const stk::mesh::MetaData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&,
    ElemDataRequests&);

  KOKKOS_DEFAULTED_FUNCTION
  ScalarEdgeOpenSolverAlg() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~ScalarEdgeOpenSolverAlg() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:
  unsigned scalarQ_          {stk::mesh::InvalidOrdinal};
  unsigned bcScalarQ_        {stk::mesh::InvalidOrdinal};
  unsigned dqdx_             {stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_    {stk::mesh::InvalidOrdinal};
  unsigned coordinates_      {stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};

  DoubleType relaxFac_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* SCALAREDGEOPENSOLVERALG_H */
