// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EffSSTDiffFluxCoeffAlg_h
#define EffSSTDiffFluxCoeffAlg_h

#include <Algorithm.h>

#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;

class EffSSTDiffFluxCoeffAlg : public Algorithm
{
public:
  using DblType = double;

  EffSSTDiffFluxCoeffAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    const double sigmaOne,
    const double sigmaTwo);

  virtual ~EffSSTDiffFluxCoeffAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* viscField_{nullptr};
  unsigned visc_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned evisc_{stk::mesh::InvalidOrdinal};
  unsigned fOneBlend_{stk::mesh::InvalidOrdinal};
  const DblType sigmaOne_;
  const DblType sigmaTwo_;
};

} // namespace nalu
} // namespace sierra

#endif
