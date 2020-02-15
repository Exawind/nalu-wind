// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.


#ifndef NODALGRADPOPENBOUNDARYALG_H
#define NODALGRADPOPENBOUNDARYALG_H

#include<Algorithm.h>
#include<ElemDataRequests.h>

#include "stk_mesh/base/Types.hpp"

namespace sierra{
namespace nalu{

template<typename AlgTraits>
class NodalGradPOpenBoundary : public Algorithm
{
public:
  NodalGradPOpenBoundary(
    Realm &,
    stk::mesh::Part *,
    const bool useShifted);

  virtual ~NodalGradPOpenBoundary() = default;

  virtual void execute() override;

  const bool useShifted_;
  const bool zeroGrad_;
  const bool massCorr_;

  const unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  const unsigned dualNodalVol_   {stk::mesh::InvalidOrdinal};
  const unsigned exposedPressureField_ {stk::mesh::InvalidOrdinal};
  const unsigned pressureField_  {stk::mesh::InvalidOrdinal};
  const unsigned gradP_          {stk::mesh::InvalidOrdinal};
  const unsigned coordinates_    {stk::mesh::InvalidOrdinal};

  MasterElement* meFC_ {nullptr};
  MasterElement* meSCS_{nullptr};

  ElemDataRequests faceData_;
  ElemDataRequests elemData_;
};

} // namespace nalu
} // namespace Sierra

#endif
