/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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
