/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MomentumAdvDiffHOElemKernel_h
#define MomentumAdvDiffHOElemKernel_h

#include <kernel/Kernel.h>
#include <KokkosInterface.h>
#include <AlgTraits.h>
#include <TimeIntegrator.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <FieldTypeDef.h>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class Realm;
class MasterElement;

template<typename AlgTraits>
class MomentumAdvDiffHOElemKernel final : public Kernel
{
  DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  MomentumAdvDiffHOElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&,
    bool);

  void setup(const TimeIntegrator& ti) { projTimeScale_ = ti.get_time_step() / ti.get_gamma1(); }
  void execute(SharedMemView<DoubleType**>&, SharedMemView<DoubleType*>&, ScratchViewsHO<DoubleType>&);
private:
  const bool reduced_sens_;


  double projTimeScale_{1.0};

  VectorFieldType* coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  VectorFieldType* Gp_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  ScalarFieldType* pressure_{nullptr};
  ScalarFieldType* density_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};
};



} // namespace nalu
} // namespace Sierra

#endif
