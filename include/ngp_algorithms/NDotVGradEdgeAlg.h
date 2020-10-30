// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef NDOTVGRADEDGEALG_H
#define NDOTVGRADEDGEALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template<typename PhiType, typename GradPhiType>
class NDotVGradEdgeAlg : public Algorithm
{
  static_assert(
    ((std::is_same<PhiType, ScalarFieldType>::value &&
      std::is_same<GradPhiType, VectorFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, GenericFieldType>::value)),
    "Improper field types passed to nodal gradient calculator");

public:
  using DblType = double;

  NDotVGradEdgeAlg(
    Realm&,
    stk::mesh::Part*,
    PhiType* phi,
    GradPhiType* gradPhi);

  virtual ~NDotVGradEdgeAlg() = default;

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

using ScalarNDotVGradEdgeAlg =
  NDotVGradEdgeAlg<ScalarFieldType, VectorFieldType>;
using VectorNDotVGradEdgeAlg =
  NDotVGradEdgeAlg<VectorFieldType, GenericFieldType>;

}  // nalu
}  // sierra


#endif /* NDOTVGRADEDGEALG_H */
