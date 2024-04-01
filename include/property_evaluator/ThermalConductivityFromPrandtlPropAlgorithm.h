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
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;
class PropertyEvaluator;

class ThermalConductivityFromPrandtlPropAlgorithm : public Algorithm
{
public:
  ThermalConductivityFromPrandtlPropAlgorithm(
    Realm& realm, const stk::mesh::PartVector& part_vec, const double Pr);

  virtual ~ThermalConductivityFromPrandtlPropAlgorithm() {}

  virtual void execute();
  const double Pr_;
};

} // namespace nalu
} // namespace sierra

#endif
