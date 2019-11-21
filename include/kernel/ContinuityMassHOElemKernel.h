// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ContinuityMassHOElemKernel_h
#define ContinuityMassHOElemKernel_h

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
class ContinuityMassHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  ContinuityMassHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs);

  void setup(const TimeIntegrator& timeIntegrator) final;

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  ScalarFieldType *densityNm1_{nullptr};
  ScalarFieldType *densityN_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};

  VectorFieldType* coordinates_{nullptr};

  double gamma_[3];

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};
};

} // namespace nalu
} // namespace Sierra

#endif
