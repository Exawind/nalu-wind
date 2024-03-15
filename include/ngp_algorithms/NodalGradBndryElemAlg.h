// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NODALGRADBNDRYELEMALG_H
#define NODALGRADBNDRYELEMALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "ngp_utils/NgpScratchData.h"
#include "ngp_algorithms/ViewHelper.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MasterElement;

using NodalGradBndryElemSimdDataType =
  sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

template <
  typename AlgTraits,
  typename PhiType,
  typename GradPhiType,
  typename ViewHelperType>
class NodalGradBndryElemAlg : public Algorithm
{
public:
  NodalGradBndryElemAlg(
    Realm&,
    stk::mesh::Part*,
    PhiType* phi,
    GradPhiType* gradPhi,
    const bool useShifted = false);

  virtual ~NodalGradBndryElemAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned phi_{stk::mesh::InvalidOrdinal};
  unsigned gradPhi_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  int phiSize_{0};
  int gradPhiSize_{0};

  const bool useShifted_{false};

  MasterElement* meFC_{nullptr};
};

template <typename AlgTraits>
using ScalarNodalGradBndryElemAlg = NodalGradBndryElemAlg<
  AlgTraits,
  ScalarFieldType,
  VectorFieldType,
  nalu_ngp::ScalarViewHelper<NodalGradBndryElemSimdDataType, ScalarFieldType>>;

template <typename AlgTraits>
using VectorNodalGradBndryElemAlg = NodalGradBndryElemAlg<
  AlgTraits,
  VectorFieldType,
  GenericFieldType,
  nalu_ngp::VectorViewHelper<NodalGradBndryElemSimdDataType, VectorFieldType>>;

template <typename AlgTraits>
using TensorNodalGradBndryElemAlg = NodalGradBndryElemAlg<
  AlgTraits,
  VectorFieldType,
  TensorFieldType,
  nalu_ngp::VectorViewHelper<NodalGradBndryElemSimdDataType, VectorFieldType>>;

} // namespace nalu
} // namespace sierra

#endif /* NODALGRADBNDRYELEMALG_H */
