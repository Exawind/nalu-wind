/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef LinearRampMeshDisplacementAuxFunction_h
#define LinearRampMeshDisplacementAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class LinearRampMeshDisplacementAuxFunction : public AuxFunction
{
public:

  LinearRampMeshDisplacementAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    std::vector<double> theParams);

  virtual ~LinearRampMeshDisplacementAuxFunction() {}
  
  using AuxFunction::do_evaluate;
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
  std::vector<double> ramp_;
};

} // namespace nalu
} // namespace Sierra

#endif
