// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DropletVelocityAuxFunction_h
#define DropletVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra {
namespace nalu {

class DropletVelocityAuxFunction : public AuxFunction
{
public:
  DropletVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double>& params);

  virtual ~DropletVelocityAuxFunction() {}

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

  double droppos_x_;
  double droppos_y_;
  double droppos_z_;
  double dropvel_x_;
  double dropvel_y_;
  double dropvel_z_;
  double radius_;
  double interface_thickness_;
};

} // namespace nalu
} // namespace sierra

#endif
