/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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

  KOKKOS_FORCEINLINE_FUNCTION
  ScalarEdgeOpenSolverAlg() = default;

  KOKKOS_FORCEINLINE_FUNCTION
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
