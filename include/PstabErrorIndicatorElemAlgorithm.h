// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef PstabErrorIndicatorElemAlgorithm_h
#define PstabErrorIndicatorElemAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class PstabErrorIndicatorElemAlgorithm : public Algorithm
{
public:

  PstabErrorIndicatorElemAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *pressure,
    VectorFieldType *Gpdx,
    const bool simpleGradApproach = false);
  ~PstabErrorIndicatorElemAlgorithm();

  void execute();

  // extract fields; nodal
  ScalarFieldType *pressure_;
  VectorFieldType *Gpdx_;
  VectorFieldType *coordinates_;
  GenericFieldType *pstabEI_;

  const double simpleGradApproachScale_;

};

} // namespace nalu
} // namespace Sierra

#endif
