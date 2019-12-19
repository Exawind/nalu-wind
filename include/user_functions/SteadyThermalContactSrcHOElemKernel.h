// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SteadyThermalContactSrcHOElemKernel_h
#define SteadyThermalContactSrcHOElemKernel_h

#include <kernel/Kernel.h>
#include <FieldTypeDef.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <memory>

namespace sierra{
namespace nalu{

class Realm;
class ElemDataRequests;

template <class AlgTraits>
class SteadyThermalContactSrcHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  SteadyThermalContactSrcHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs);

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  VectorFieldType *coordinates_{nullptr};
  const double a_{1.0};
  const double k_{1.0};

  CVFEMOperators<AlgTraits::polyOrder_> ops_;
};

} // namespace nalu
} // namespace Sierra

#endif
