/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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
