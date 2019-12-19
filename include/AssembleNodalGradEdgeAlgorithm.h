// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodalGradEdgeAlgorithm_h
#define AssembleNodalGradEdgeAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;
class AssembleNodalGradEdgeAlgorithm : public Algorithm
{
public:

  AssembleNodalGradEdgeAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *scalarQ,
    VectorFieldType *dqdx);
  virtual ~AssembleNodalGradEdgeAlgorithm() {}

  virtual void execute();

  ScalarFieldType *scalarQ_;
  VectorFieldType *dqdx_;
  VectorFieldType *edgeAreaVec_;
  ScalarFieldType *dualNodalVolume_;

};

} // namespace nalu
} // namespace Sierra

#endif
