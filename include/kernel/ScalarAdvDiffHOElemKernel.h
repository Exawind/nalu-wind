/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarAdvDiffHOElemKernel_h
#define ScalarAdvDiffHOElemKernel_h

#include <kernel/Kernel.h>
#include <KokkosInterface.h>
#include <AlgTraits.h>
#include <TimeIntegrator.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class Realm;
class MasterElement;

template<typename AlgTraits>
class ScalarAdvDiffHOElemKernel final : public Kernel
{
  DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  ScalarAdvDiffHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests& dataPreReqs);

  ~ScalarAdvDiffHOElemKernel() = default;

  void setup(const TimeIntegrator& ti) final { projTimeScale_ = ti.get_time_step() / ti.get_gamma1(); }

  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  ScalarFieldType *scalarQ_{nullptr};
  ScalarFieldType *diffFluxCoeff_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  VectorFieldType *velocity_{nullptr};
  VectorFieldType* Gp_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  ScalarFieldType* pressure_{nullptr};
  ScalarFieldType* density_{nullptr};

  double projTimeScale_{1.0};

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};

};



} // namespace nalu
} // namespace Sierra

#endif
