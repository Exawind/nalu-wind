// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef EnthalpyEffDiffFluxCoeffAlg_h
#define EnthalpyEffDiffFluxCoeffAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra{
namespace nalu{

class Realm;

class EnthalpyEffDiffFluxCoeffAlg : public Algorithm
{
public:
  using DblType = double;

  EnthalpyEffDiffFluxCoeffAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    const double sigmaTurb,
    const bool isTurbulent);

  virtual ~EnthalpyEffDiffFluxCoeffAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* specHeatField_ {nullptr};
  unsigned thermalCond_ {stk::mesh::InvalidOrdinal};
  unsigned specHeat_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned evisc_ {stk::mesh::InvalidOrdinal};
  const DblType invSigmaTurb_;  
  const bool isTurbulent_;  
};

} // namespace nalu
} // namespace Sierra

#endif
