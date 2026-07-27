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
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "ngp_utils/NgpScratchData.h"
#include "ngp_algorithms/ViewHelper.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra::kynema_ugf {

class MasterElement;

using FluxDivBndryElemSimdDataType =
  sierra::kynema_ugf::kynema_ugf_ngp::ElemSimdData<stk::mesh::NgpMesh>;

template <typename AlgTraits>
class FluxDivBndryElemAlg final : public Algorithm
{
public:
  FluxDivBndryElemAlg(
    Realm& realm,
    stk::mesh::Part* part,
    GenericFieldType* exposed_flux,
    ScalarFieldType* div_flux);
  virtual void execute() final;

private:
  ElemDataRequests dataNeeded_;

  unsigned flux_{stk::mesh::InvalidOrdinal};
  unsigned div_flux_{stk::mesh::InvalidOrdinal};
  unsigned dnv_{stk::mesh::InvalidOrdinal};
  MasterElement* meFC_{nullptr};
};

} // namespace sierra::kynema_ugf
