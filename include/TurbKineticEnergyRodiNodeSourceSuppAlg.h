/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyRodiNodeSourceSuppAlg_h
#define TurbKineticEnergyRodiNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TurbKineticEnergyRodiNodeSourceSuppAlg : public SupplementalAlgorithm
{
public:

  TurbKineticEnergyRodiNodeSourceSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergyRodiNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);

  VectorFieldType *dhdx_;
  ScalarFieldType *specificHeat_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *dualNodalVolume_;
  const double beta_;
  const double turbPr_;
  const int nDim_;
  std::vector<double> gravity_;
};

} // namespace nalu
} // namespace Sierra

#endif
