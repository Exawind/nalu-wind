/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbViscSSTTAMSAlgorithm_h
#define TurbViscSSTTAMSAlgorithm_h

#include <Algorithm.h>

#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;

class TurbViscSSTTAMSAlgorithm : public Algorithm
{
public:
  TurbViscSSTTAMSAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~TurbViscSSTTAMSAlgorithm() {}
  virtual void execute();

  const double aOne_;
  const double betaStar_;

  ScalarFieldType* density_;
  ScalarFieldType* viscosity_;
  ScalarFieldType* tke_;
  ScalarFieldType* sdr_;
  ScalarFieldType* minDistance_;
  GenericFieldType* dudx_;
  ScalarFieldType* tvisc_;
};

} // namespace nalu
} // namespace sierra

#endif
