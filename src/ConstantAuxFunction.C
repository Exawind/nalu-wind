// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ConstantAuxFunction.h>
#include <algorithm>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {

ConstantAuxFunction::ConstantAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double>& values)
  : AuxFunction(beginPos, endPos), values_(values)
{
  STK_ThrowRequire(endPos_ <= values_.size());
}

void
ConstantAuxFunction::do_evaluate(
  const double* /*coords*/,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned beginPos,
  const unsigned endPos) const
{
  const double* const vals = &values_[0];
  for (unsigned p = 0; p < numPoints; ++p) {
    for (unsigned i = beginPos; i < endPos; ++i) {
      fieldPtr[i] = vals[i];
    }
    fieldPtr += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
