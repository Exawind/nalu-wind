/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TAMSAlgDriver_h
#define TAMSAlgDriver_h

#include <FieldTypeDef.h>
#include "ngp_algorithms/NgpAlgDriver.h"
#include "ngp_algorithms/FieldUpdateAlgDriver.h"
#include "ngp_algorithms/TAMSAvgMdotEdgeAlg.h"
#include "ngp_algorithms/TAMSAvgMdotElemAlg.h"
#include "ngp_algorithms/SSTTAMSAveragesAlg.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class Realm;

class TAMSAlgDriver
{

public:
  using DblType = double;

  TAMSAlgDriver(Realm& realm);
  virtual ~TAMSAlgDriver() = default;

  void register_fields_and_algorithms(
    stk::mesh::Part* part, const stk::topology& theTopo);
  void compute_metric_tensor();
  void compute_averages();
  void compute_avgMdot();
  void predict_state();

private:
  Realm& realm_;

  VectorFieldType* avgVelocity_;
  VectorFieldType* avgVelocityRTM_;
  ScalarFieldType* avgTkeResolved_;
  GenericFieldType* avgDudx_;
  GenericFieldType* metric_;
  ScalarFieldType* alpha_;

  ScalarFieldType* resAdequacy_;
  ScalarFieldType* avgResAdequacy_;
  ScalarFieldType* avgProduction_;
  ScalarFieldType* avgTime_;
  GenericFieldType* avgMdotScs_;
  ScalarFieldType* avgMdot_;

  bool isInit_;
  FieldUpdateAlgDriver metricTensorAlgDriver_;
  std::unique_ptr<SSTTAMSAveragesAlg> avgAlg_;
  NgpAlgDriver avgMdotAlg_;

  const TurbulenceModel turbulenceModel_;

  bool resetTAMSAverages_;
};
} // namespace nalu
} // namespace sierra

#endif
