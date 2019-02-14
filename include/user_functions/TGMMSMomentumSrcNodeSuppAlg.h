/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TGMMSMomentumSrcNodeSuppAlg_h
#define TGMMSMomentumSrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TGMMSMomentumSrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  TGMMSMomentumSrcNodeSuppAlg(Realm &realm);
  void node_execute(double *lhs, double *rhs, stk::mesh::Entity node);
private:
  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;
  static constexpr double mu{1.0e-3};
};

} // namespace nalu
} // namespace Sierra

#endif
