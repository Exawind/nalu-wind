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

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MasterElement;

template <typename AlgTraits, typename PhiType, typename GradPhiType>
class NodalGradBndryElemAlg : public Algorithm
{
  static_assert(
    ((std::is_same<PhiType, ScalarFieldType>::value &&
      std::is_same<GradPhiType, VectorFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, GenericFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, TensorFieldType>::value)),
    "Improper field types passed to nodal gradient calculator");

public:
  using DblType = double;

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

  const bool useShifted_{false};

  MasterElement* meFC_{nullptr};

  static constexpr int NumComp =
    std::is_same<PhiType, ScalarFieldType>::value ? 1 : AlgTraits::nDim_;
};

template <typename AlgTraits>
using ScalarNodalGradBndryElemAlg =
  NodalGradBndryElemAlg<AlgTraits, ScalarFieldType, VectorFieldType>;

template <typename AlgTraits>
using VectorNodalGradBndryElemAlg =
  NodalGradBndryElemAlg<AlgTraits, VectorFieldType, GenericFieldType>;
template <typename AlgTraits>
using TensorNodalGradBndryElemAlg =
  NodalGradBndryElemAlg<AlgTraits, VectorFieldType, TensorFieldType>;

} // namespace nalu
} // namespace sierra

#endif /* NODALGRADBNDRYELEMALG_H */
