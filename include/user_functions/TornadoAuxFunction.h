// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TornadoAuxFunction_h
#define TornadoAuxFunction_h

#include <AuxFunction.h>

namespace sierra {
namespace nalu {

class TornadoAuxFunction : public AuxFunction
{
public:
  TornadoAuxFunction(const unsigned beginPos, const unsigned endPos);

  virtual ~TornadoAuxFunction() {}

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
  const double z1_, hNot_, rNot_, uRef_, swirl_;
};

} // namespace nalu
} // namespace sierra

#endif
