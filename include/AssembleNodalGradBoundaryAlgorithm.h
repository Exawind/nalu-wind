// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodalGradBoundaryAlgorithm_h
#define AssembleNodalGradBoundaryAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradBoundaryAlgorithm : public Algorithm
{
public:
  AssembleNodalGradBoundaryAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *scalarQ,
    VectorFieldType *dqdx,
    const bool useShifted);
  virtual ~AssembleNodalGradBoundaryAlgorithm() {}

  virtual void execute();

  ScalarFieldType *scalarQ_;
  VectorFieldType *dqdx_;
  const bool useShifted_;

};

} // namespace nalu
} // namespace Sierra

#endif
