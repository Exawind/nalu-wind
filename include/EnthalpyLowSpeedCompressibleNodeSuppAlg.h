// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef EnthalpyLowSpeedCompressibleNodeSuppAlg_h
#define EnthalpyLowSpeedCompressibleNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class EnthalpyLowSpeedCompressibleNodeSuppAlg : public SupplementalAlgorithm
{
public:

  EnthalpyLowSpeedCompressibleNodeSuppAlg(
    Realm &realm);

  virtual ~EnthalpyLowSpeedCompressibleNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  ScalarFieldType *pressureN_;
  ScalarFieldType *pressureNp1_;
  ScalarFieldType *dualNodalVolume_;
  double dt_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
