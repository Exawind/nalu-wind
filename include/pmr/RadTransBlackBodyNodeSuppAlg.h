// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef RadTransBlackBodyNodeSuppAlg_h
#define RadTransBlackBodyNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class RadiativeTransportEquationSystem;
class Realm;

class RadTransBlackBodyNodeSuppAlg : public SupplementalAlgorithm
{
public:

  RadTransBlackBodyNodeSuppAlg(
      Realm &realm,
      RadiativeTransportEquationSystem *radEqSystem);

  virtual ~RadTransBlackBodyNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
      double *lhs,
      double *rhs,
      stk::mesh::Entity node);
 
  const RadiativeTransportEquationSystem *radEqSystem_;
  ScalarFieldType *intensity_;
  ScalarFieldType *absorption_;
  ScalarFieldType *scattering_;
  ScalarFieldType *radiationSource_;
  ScalarFieldType *dualNodalVolume_;

};

} // namespace nalu
} // namespace Sierra

#endif
