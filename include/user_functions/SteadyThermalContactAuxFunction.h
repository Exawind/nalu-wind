// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SteadyThermalContactAuxFunction_h
#define SteadyThermalContactAuxFunction_h

#include <AuxFunction.h>

#include "stk_mesh/base/NgpField.hpp"

#include <vector>

namespace YAML { class Node; }

namespace sierra{
namespace nalu{

struct SteadyThermalContactData
{
  const double wave_number{1.0};
  const double amplitude{0.25};
  const stk::mesh::NgpField<double> coordinate_field;
  mutable stk::mesh::NgpField<double> temperature_field;
};

void execute(SteadyThermalContactData& data, const stk::mesh::FastMeshIndex& mi);

class SteadyThermalContactAuxFunction : public AuxFunction
{
public:

  SteadyThermalContactAuxFunction();

  virtual ~SteadyThermalContactAuxFunction() {}
  
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
  double a_;
  double k_;
  double pi_;

};

} // namespace nalu
} // namespace Sierra

#endif
