/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ConstantAuxFunction_h
#define ConstantAuxFunction_h

#include <AuxFunction.h>
#include <vector>

#include<vector>

namespace sierra{
namespace nalu{

class ConstantAuxFunction : public AuxFunction
{
public:
  using AuxFunction::do_evaluate;

  ConstantAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double> & values);

  virtual ~ConstantAuxFunction() {}
  
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
  
private:
  const std::vector<double> values_;
};

} // namespace nalu
} // namespace Sierra

#endif
