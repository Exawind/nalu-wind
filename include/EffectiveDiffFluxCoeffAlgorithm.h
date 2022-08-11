// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EffectiveDiffFluxCoeffAlgorithm_h
#define EffectiveDiffFluxCoeffAlgorithm_h

#include <Algorithm.h>

#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;

class EffectiveDiffFluxCoeffAlgorithm : public Algorithm
{
public:
  EffectiveDiffFluxCoeffAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    ScalarFieldType* visc,
    ScalarFieldType* tvisc,
    ScalarFieldType* evisc,
    const double sigmaLam,
    const double sigmaTurb);
  virtual ~EffectiveDiffFluxCoeffAlgorithm() {}
  virtual void execute();

  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* evisc_;

  const double sigmaLam_;
  const double sigmaTurb_;
  const bool isTurbulent_;
};

} // namespace nalu
} // namespace sierra

#endif
