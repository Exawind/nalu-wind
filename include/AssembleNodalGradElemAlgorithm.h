// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodalGradElemAlgorithm_h
#define AssembleNodalGradElemAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradElemAlgorithm : public Algorithm
{
public:

  AssembleNodalGradElemAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *scalarQ,
    VectorFieldType *dqdx,
    const bool useShifted = false);
  virtual ~AssembleNodalGradElemAlgorithm() {}

  virtual void execute();

  ScalarFieldType *scalarQ_;
  VectorFieldType *dqdx_;
  ScalarFieldType *dualNodalVolume_;
  VectorFieldType *coordinates_;

  const bool useShifted_;
};

} // namespace nalu
} // namespace Sierra

#endif
