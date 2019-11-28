// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MomentumBuoyancySrcHOElemKernel_h
#define MomentumBuoyancySrcHOElemKernel_h

#include <kernel/Kernel.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

// Kokkos
#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class TimeIntegrator;

template<class AlgTraits>
class MomentumBuoyancySrcHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  MomentumBuoyancySrcHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs);

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  const double rhoRef_;
  const std::array<double,3> gravity_;

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* density_{nullptr};


  CVFEMOperators<AlgTraits::polyOrder_> ops_;
};

} // namespace nalu
} // namespace Sierra

#endif
