// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodalGradPOpenBoundaryAlgorithm_h
#define AssembleNodalGradPOpenBoundaryAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradPOpenBoundaryAlgorithm : public Algorithm
{
public:
  AssembleNodalGradPOpenBoundaryAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    const bool useShifted);
  virtual ~AssembleNodalGradPOpenBoundaryAlgorithm() {}

  virtual void execute();

  const bool useShifted_;
  const bool zeroGrad_;
  const bool massCorr_;
};

} // namespace nalu
} // namespace Sierra

#endif
