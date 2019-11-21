// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef PressurePoissonHOElemKernel_h
#define PressurePoissonHOElemKernel_h

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
class PressurePoissonHOElemKernel final : public Kernel
{
  using ViewTypes = CVFEMViews<AlgTraits::polyOrder_>;
  DeclareCVFEMTypeDefs(ViewTypes);

public:
  PressurePoissonHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs,
    bool reduced_sens);

  void setup(const TimeIntegrator& ti) { projTimeScale_ = ti.get_time_step() / ti.get_gamma1(); }
  using Kernel::execute;
  void execute(SharedMemView<DoubleType**>&, SharedMemView<DoubleType*>&, ScratchViewsHO<DoubleType>&);

private:
  const bool reduced_sens_;
  double projTimeScale_{100.0};

  VectorFieldType* coordinates_;
  ScalarFieldType* pressure_;
  VectorFieldType* Gp_;
  ScalarFieldType* density_;
  VectorFieldType* velocity_;

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};
};



} // namespace nalu
} // namespace Sierra

#endif
