// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TurbViscSSTAlg_h
#define TurbViscSSTAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class TurbViscSSTAlg : public Algorithm
{
public:
  using DblType = double;

  TurbViscSSTAlg(
    Realm& realm,
    stk::mesh::Part* part,
    ScalarFieldType* tvisc,
    const bool = false);

  virtual ~TurbViscSSTAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* tviscField_{nullptr};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned tke_{stk::mesh::InvalidOrdinal};
  unsigned sdr_{stk::mesh::InvalidOrdinal};
  unsigned minDistance_{stk::mesh::InvalidOrdinal};
  unsigned dwalldistdx_{stk::mesh::InvalidOrdinal};
  unsigned dnDotVdx_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned dpdx_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};

  const DblType aOne_;
  const DblType betaStar_;
  const DblType sThres_;
};

} // namespace nalu
} // namespace sierra

#endif
