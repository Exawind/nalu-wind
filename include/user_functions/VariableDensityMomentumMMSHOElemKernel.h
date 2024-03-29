// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VariableDensityMomentumMMSHOElemKernel_h
#define VariableDensityMomentumMMSHOElemKernel_h

#include <kernel/Kernel.h>
#include <FieldTypeDef.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <memory>

namespace sierra {
namespace nalu {

class Realm;
class ElemDataRequests;

template <class AlgTraits>
class VariableDensityMomentumMMSHOElemKernel final : public Kernel
{
  DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);

public:
  VariableDensityMomentumMMSHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs);

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  const double unot_;
  const double vnot_;
  const double wnot_;
  const double pnot_;
  const double hnot_;
  const double a_;
  const double ah_;
  const double visc_;
  const double Pref_;
  const double MW_;
  const double R_;
  const double Tref_;
  const double Cp_;
  const double pi_;
  const double twoThirds_;
  double rhoRef_;
  double gx_;
  double gy_;
  double gz_;

  VectorFieldType* coordinates_{nullptr};
  CVFEMOperators<AlgTraits::polyOrder_> ops_;
};

} // namespace nalu
} // namespace sierra

#endif
