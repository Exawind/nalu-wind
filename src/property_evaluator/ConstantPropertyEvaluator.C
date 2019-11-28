// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <property_evaluator/ConstantPropertyEvaluator.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ConstantPropertyEvaluator
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ConstantPropertyEvaluator::ConstantPropertyEvaluator(const double & value)
  : value_(value)
{
  // nothing else
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ConstantPropertyEvaluator::~ConstantPropertyEvaluator()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
double
ConstantPropertyEvaluator::execute(
  double * /* indVarList */,
  stk::mesh::Entity /*node*/)
{
  return value_;
}

} // namespace nalu
} // namespace Sierra


