// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ThermalConductivityFromPrandtlPropAlgorithm_h
#define ThermalConductivityFromPrandtlPropAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
class FieldBase;
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;
class PropertyEvaluator;

class ThermalConductivityFromPrandtlPropAlgorithm : public Algorithm
{
public:

  ThermalConductivityFromPrandtlPropAlgorithm(
    Realm & realm,
    stk::mesh::Part * part,
    ScalarFieldType *thermalCond,
    ScalarFieldType *specificHeat,
    ScalarFieldType *viscosity,
    const double Pr);

  virtual ~ThermalConductivityFromPrandtlPropAlgorithm() {}

  virtual void execute();
  
  ScalarFieldType *thermalCond_;
  ScalarFieldType *specHeat_;
  ScalarFieldType *viscosity_;

  const double Pr_;
};

} // namespace nalu
} // namespace Sierra

#endif
