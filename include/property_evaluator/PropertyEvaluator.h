// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PropertyEvaluator_h
#define PropertyEvaluator_h

#include <stk_mesh/base/Entity.hpp>

#include <vector>

namespace sierra {
namespace nalu {

class PropertyEvaluator
{
public:
  PropertyEvaluator() {}
  virtual ~PropertyEvaluator() {}

  virtual double
  execute(double* indVarList, stk::mesh::Entity node = stk::mesh::Entity()) = 0;
};

} // namespace nalu
} // namespace sierra

#endif
