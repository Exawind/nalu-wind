// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#pragma once

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra::kynema_ugf {

class FluxDivEdgeAlg final : public Algorithm
{
public:
  using DblType = double;
  FluxDivEdgeAlg(
    Realm&, stk::mesh::Part*, ScalarFieldType* flux, ScalarFieldType* div_flux);
  virtual void execute() final;

private:
  unsigned flux_{stk::mesh::InvalidOrdinal};
  unsigned div_flux_{stk::mesh::InvalidOrdinal};
  unsigned dnv_{stk::mesh::InvalidOrdinal};
};

} // namespace sierra::kynema_ugf
