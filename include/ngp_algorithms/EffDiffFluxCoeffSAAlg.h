// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EFFDIFFFLUXCOEFFSAALG_H
#define EFFDIFFFLUXCOEFFSAALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

class Realm;

/** Compute effective diffusive flux coefficient for Spalart-Allmaras
 *
 *  For the SA model, the diffusion term in the nuTilda equation is:
 *    (1/sigma) * div[ (mu + rho*nuTilda) * grad(nuTilda) ]
 *
 *  Therefore the effective diffusivity is:
 *    evisc = (mu + rho * nuTilda * fn) / sigma
 *  with fn = 1 for nuTilda >= 0 and the SA-neg fn otherwise.
 *
 *  This differs from the standard EffDiffFluxCoeffAlg which computes
 *    evisc = mu/sigma_lam + mu_t/sigma_turb
 *  because mu_t = rho*nuTilda*fv1 != rho*nuTilda in general.
 */
class EffDiffFluxCoeffSAAlg : public Algorithm
{
public:
  using DblType = double;

  EffDiffFluxCoeffSAAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType* visc,
    ScalarFieldType* evisc,
    const double sigma);

  virtual ~EffDiffFluxCoeffSAAlg() = default;

  virtual void execute() override;

private:
  //! For use within selectField to determine selector
  ScalarFieldType* viscField_{nullptr};

  //! Laminar viscosity field
  unsigned visc_{stk::mesh::InvalidOrdinal};

  //! Density field
  unsigned density_{stk::mesh::InvalidOrdinal};

  //! SA working variable nuTilda
  unsigned nuTilda_{stk::mesh::InvalidOrdinal};

  //! Effective viscosity used in diffusion terms
  unsigned evisc_{stk::mesh::InvalidOrdinal};

  //! 1/sigma
  const DblType invSigma_;

  //! SA-neg constant for fn in the negative branch
  const DblType Cn1_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif /* EFFDIFFFLUXCOEFFSAALG_H */
