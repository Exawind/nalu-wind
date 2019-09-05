/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NODALGRADEDGEALG_H
#define NODALGRADEDGEALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template<typename PhiType, typename GradPhiType>
class NodalGradEdgeAlg : public Algorithm
{
  static_assert(
    ((std::is_same<PhiType, ScalarFieldType>::value &&
      std::is_same<GradPhiType, VectorFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, GenericFieldType>::value)),
    "Improper field types passed to nodal gradient calculator");

public:
  using DblType = double;

  NodalGradEdgeAlg(
    Realm&,
    stk::mesh::Part*,
    PhiType* phi,
    GradPhiType* gradPhi);

  virtual ~NodalGradEdgeAlg() = default;

  virtual void execute() override;

private:
  unsigned phi_ {stk::mesh::InvalidOrdinal};
  unsigned gradPhi_ {stk::mesh::InvalidOrdinal};

  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_ {stk::mesh::InvalidOrdinal};

  //! Number of components (ScalarFieldType = 1; VectorFieldType = nDim)
  const int dim1_;
  //! Spatial dimension (2D or 3D)
  const int dim2_;

  //! Maximum size for static arrays used within device loops
  static constexpr int NDimMax = 3;
};

using ScalarNodalGradEdgeAlg =
  NodalGradEdgeAlg<ScalarFieldType, VectorFieldType>;
using VectorNodalGradEdgeAlg =
  NodalGradEdgeAlg<VectorFieldType, GenericFieldType>;

}  // nalu
}  // sierra


#endif /* NODALGRADEDGEALG_H */
