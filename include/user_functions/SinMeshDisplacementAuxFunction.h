/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SinMeshDisplacementAuxFunction_h
#define SinMeshDisplacementAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class SinMeshDisplacementAuxFunction : public AuxFunction
{
public:

  SinMeshDisplacementAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    std::vector<double> theParams);

  virtual ~SinMeshDisplacementAuxFunction() {}
  
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
  double pi_;
  double maxDisplacement_;

};

} // namespace nalu
} // namespace Sierra

#endif
