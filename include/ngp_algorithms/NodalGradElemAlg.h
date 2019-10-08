/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NODALGRADELEMALG_H
#define NODALGRADELEMALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MasterElement;

template<typename AlgTraits, typename PhiType, typename GradPhiType>
class NodalGradElemAlg : public Algorithm
{
  static_assert(
    ((std::is_same<PhiType, ScalarFieldType>::value &&
      std::is_same<GradPhiType, VectorFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, GenericFieldType>::value)),
    "Improper field types passed to nodal gradient calculator");

public:
  using DblType = double;

  NodalGradElemAlg(
    Realm&,
    stk::mesh::Part*,
    PhiType* phi,
    GradPhiType* gradPhi,
    const bool useShifted = false);

  virtual ~NodalGradElemAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned phi_{stk::mesh::InvalidOrdinal};
  unsigned gradPhi_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_ {stk::mesh::InvalidOrdinal};

  const bool useShifted_{false};

  MasterElement* meSCS_{nullptr};

  static constexpr int NDimMax = 3;

  static constexpr int NumComp =
    std::is_same<PhiType, ScalarFieldType>::value ? 1 : AlgTraits::nDim_;
};

template <typename AlgTraits>
using ScalarNodalGradElemAlg =
  NodalGradElemAlg<AlgTraits, ScalarFieldType, VectorFieldType>;

template<typename AlgTraits>
using VectorNodalGradElemAlg =
  NodalGradElemAlg<AlgTraits,VectorFieldType, GenericFieldType>;

}  // nalu
}  // sierra


#endif /* NODALGRADELEMALG_H */
