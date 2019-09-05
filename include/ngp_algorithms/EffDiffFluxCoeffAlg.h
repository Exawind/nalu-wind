/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef EFFDIFFFLUXCOEFFALG_H
#define EFFDIFFFLUXCOEFFALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

/** Compute effective diffusive flux coefficient
 */
class EffDiffFluxCoeffAlg : public Algorithm
{
public:
  using DblType = double;

  EffDiffFluxCoeffAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    const double sigmaLam,
    const double sigmaTurb,
    const bool isTurbulent);

  virtual ~EffDiffFluxCoeffAlg() = default;

  virtual void execute() override;

private:
  // For use within selectField to determine selector
  ScalarFieldType* viscField_ {nullptr};

  //! Laminar viscosity field
  unsigned visc_  {stk::mesh::InvalidOrdinal};

  //! Turbulent viscosity field (computed in TurbVisc Algorithms)
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};

  //! Effective viscosity used in diffusion terms
  unsigned evisc_ {stk::mesh::InvalidOrdinal};

  //! reciprocal of the laminar sigma coefficient
  const DblType invSigmaLam_;

  //! reciprocal of the turbulent sigma coefficient
  const DblType invSigmaTurb_;

  //! Flag indicating whether a turbulence model is active
  const bool isTurbulent_;
};

}  // nalu
}  // sierra


#endif /* EFFDIFFFLUXCOEFFALG_H */
