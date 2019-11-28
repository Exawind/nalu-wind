// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MomentumMassHOElemKernel_h
#define MomentumMassHOElemKernel_h

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

template<class AlgTraits>
class MomentumMassHOElemKernel final : public Kernel
{
  using ViewTypes = CVFEMViews<AlgTraits::polyOrder_>;
  DeclareCVFEMTypeDefs(ViewTypes);

public:
  MomentumMassHOElemKernel(
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
  VectorFieldType *velocityNm1_{nullptr};
  VectorFieldType *velocityN_{nullptr};
  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *densityNm1_{nullptr};
  ScalarFieldType *densityN_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};

  VectorFieldType* coordinates_{nullptr};
  VectorFieldType* Gp_{nullptr};

  double gamma_[3];

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};
};

} // namespace nalu
} // namespace Sierra

#endif
