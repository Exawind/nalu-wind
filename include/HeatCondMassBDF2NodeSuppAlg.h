// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef HeatCondMassBDF2NodeSuppAlg_h
#define HeatCondMassBDF2NodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class HeatCondMassBDF2NodeSuppAlg : public SupplementalAlgorithm
{
public:

  HeatCondMassBDF2NodeSuppAlg(
    Realm &realm);

  virtual ~HeatCondMassBDF2NodeSuppAlg() {}

  virtual void setup();
  
  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);

  ScalarFieldType *temperatureNm1_;
  ScalarFieldType *temperatureN_;
  ScalarFieldType *temperatureNp1_;
  ScalarFieldType *density_;
  ScalarFieldType *specificHeat_;
  ScalarFieldType *dualNodalVolume_;
  double dt_;
  double gamma1_, gamma2_, gamma3_;

};

} // namespace nalu
} // namespace Sierra

#endif
