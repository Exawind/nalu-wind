/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ConstantAuxFunction.h>
#include <algorithm>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra{
namespace nalu{

ConstantAuxFunction::ConstantAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double> & values) :
  AuxFunction(beginPos, endPos),
  values_(values)
{
  ThrowRequire(endPos_ <= values_.size());
}


void
ConstantAuxFunction::do_evaluate(
  const double * /*coords*/,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned beginPos,
  const unsigned endPos) const
{
  const double * const vals = &values_[0];
  for(unsigned p=0; p < numPoints; ++p) {
    for(unsigned i=beginPos; i < endPos; ++i) {
      fieldPtr[i] = vals[i];
    }
    fieldPtr += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
