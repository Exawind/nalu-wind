// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TurbViscSAAlg_h
#define TurbViscSAAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

class Realm;

class TurbViscSAAlg : public Algorithm
{
public:
  using DblType = double;

  TurbViscSAAlg(Realm& realm, stk::mesh::Part* part, ScalarFieldType* tvisc);

  virtual ~TurbViscSAAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* tviscField_{nullptr};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned nuTilda_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};

  const DblType Cv1_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif
