// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ConstantPropertyEvaluator_h
#define ConstantPropertyEvaluator_h

#include <property_evaluator/PropertyEvaluator.h>

namespace stk {
namespace mesh {
struct Entity;
}
} // namespace stk

namespace sierra {
namespace nalu {

class ConstantPropertyEvaluator : public PropertyEvaluator
{
public:
  ConstantPropertyEvaluator(const double& value);
  virtual ~ConstantPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double value_;
};

} // namespace nalu
} // namespace sierra

#endif
