// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NODALGRADEDGEALG_H
#define NODALGRADEDGEALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template <typename PhiType, typename GradPhiType>
class NodalGradEdgeAlg : public Algorithm
{
public:
  using DblType = double;

  NodalGradEdgeAlg(
    Realm&, stk::mesh::Part*, PhiType* phi, GradPhiType* gradPhi);

  virtual ~NodalGradEdgeAlg() = default;

  virtual void execute() override;

private:
  unsigned phi_{stk::mesh::InvalidOrdinal};
  unsigned gradPhi_{stk::mesh::InvalidOrdinal};

  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_{stk::mesh::InvalidOrdinal};

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
using TensorNodalGradEdgeAlg =
  NodalGradEdgeAlg<VectorFieldType, TensorFieldType>;

} // namespace nalu
} // namespace sierra

#endif /* NODALGRADEDGEALG_H */
