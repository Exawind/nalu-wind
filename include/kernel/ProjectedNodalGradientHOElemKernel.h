// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ProjectedNodalGradientHOElemKernel_h
#define ProjectedNodalGradientHOElemKernel_h

#include <kernel/Kernel.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>

#include <CVFEMTypeDefs.h>
#include <FieldTypeDef.h>

#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;

template<class AlgTraits>
class ProjectedNodalGradientHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  ProjectedNodalGradientHOElemKernel(
    const stk::mesh::BulkData&,
    SolutionOptions&,
    std::string, std::string,
    ElemDataRequests&);

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  VectorFieldType* coordinates_{nullptr};

  ScalarFieldType* q_{nullptr};
  VectorFieldType* dqdx_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};
};

} // namespace nalu
} // namespace Sierra

#endif
