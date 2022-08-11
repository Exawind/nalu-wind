// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef OneTwoTenVelocityAuxFunction_h
#define OneTwoTenVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra {
namespace nalu {

class OneTwoTenVelocityAuxFunction : public AuxFunction
{
public:
  OneTwoTenVelocityAuxFunction(const unsigned beginPos, const unsigned endPos);

  virtual ~OneTwoTenVelocityAuxFunction() {}

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

private:
  const double dpdz_;
  const double mu_;
  const double a_;
  const double b_;
  const int maxN_;
  const double pi_;
};

} // namespace nalu
} // namespace sierra

#endif
