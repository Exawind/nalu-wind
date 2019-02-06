/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTESSTTAMSAVERAGESNODEALGORITHM_H
#define COMPUTESSTTAMSAVERAGESNODEALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <ngp_utils/NgpTypes.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeSSTTAMSAveragesNodeAlgorithm : public Algorithm
{
public:
  ComputeSSTTAMSAveragesNodeAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~ComputeSSTTAMSAveragesNodeAlgorithm() {}

  virtual void execute();

  const double betaStar_;
  const double CMdeg_;
  const bool meshMotion_;

  VectorFieldType* velocityRTM_{nullptr};
  ScalarFieldType* pressure_{nullptr};
  ScalarFieldType* density_{nullptr};
  ScalarFieldType* specDissipationRate_{nullptr};
  ScalarFieldType* turbKineticEnergy_{nullptr};
  GenericFieldType* dudx_{nullptr};
  VectorFieldType* avgVelocity_{nullptr};
  ScalarFieldType* avgPress_{nullptr};
  ScalarFieldType* avgDensity_{nullptr};
  ScalarFieldType* avgTkeRes_{nullptr};
  ScalarFieldType* avgTime_{nullptr};
  GenericFieldType* avgDudx_{nullptr};
  ScalarFieldType* avgProd_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  ScalarFieldType* alpha_{nullptr};
  ScalarFieldType* resAdeq_{nullptr};
  ScalarFieldType* avgResAdeq_{nullptr};
  GenericFieldType* Mij_{nullptr};

  std::vector<double> tauSGET;
  std::vector<double> tauSGRS;
  std::vector<double> tau;
  std::vector<double> Psgs;
};

} // namespace nalu
} // namespace sierra

#endif
