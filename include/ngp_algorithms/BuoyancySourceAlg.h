// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BUOYANCYSOURCEALG_H
#define BUOYANCYSOURCEALG_H

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class BuoyancySourceAlg : public Algorithm
{

public:
  using DblType = double;

  BuoyancySourceAlg(
    Realm&, stk::mesh::Part*, VectorFieldType* source);

  virtual ~BuoyancySourceAlg() = default;

  virtual void execute() override;

private:
  unsigned source_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};


  //! Maximum size for static arrays used within device loops
  static constexpr int NDimMax = 3;
};

} // namespace nalu
} // namespace sierra

#endif /* NODALGRADEDGEALG_H */
