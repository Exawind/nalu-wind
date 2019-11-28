// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
