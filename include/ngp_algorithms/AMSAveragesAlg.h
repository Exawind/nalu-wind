// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AMSAveragesAlg_h
#define AMSAveragesAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class AMSAveragesAlg : public Algorithm
{
public:
  using DblType = double;

  AMSAveragesAlg(Realm& realm, stk::mesh::Part* part);

  virtual ~AMSAveragesAlg() = default;

  virtual void execute() = 0;
};

} // namespace nalu
} // namespace sierra

#endif
