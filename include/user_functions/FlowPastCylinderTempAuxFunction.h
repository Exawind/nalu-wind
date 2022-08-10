// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FlowPastCylinderTempAuxFunction_h
#define FlowPastCylinderTempAuxFunction_h

#include <AuxFunction.h>

namespace sierra {
namespace nalu {

class FlowPastCylinderTempAuxFunction : public AuxFunction
{
public:
  FlowPastCylinderTempAuxFunction();

  virtual ~FlowPastCylinderTempAuxFunction() {}

  using AuxFunction::do_evaluate;
  virtual void do_evaluate(
    const double* coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double* fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;

  int find_index(const double z, int iMin, int iMax) const;
  double interpolate_data(const double z) const;
  double
  local_interpolation(const double z, const int index0, const int index1) const;

private:
  double h_;
  double k_;
  double pi_;
  double experimentalData_[25][2];
  int iMin_;
  int iMax_;
};

} // namespace nalu
} // namespace sierra

#endif
