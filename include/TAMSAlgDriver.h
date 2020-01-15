// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TAMSAlgDriver_h
#define TAMSAlgDriver_h

#include "FieldTypeDef.h"
#include "Algorithm.h"
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
  virtual void register_nodal_fields(stk::mesh::Part* part);
  virtual void
  register_element_fields(stk::mesh::Part* part, const stk::topology& theTopo);
  virtual void register_edge_fields(stk::mesh::Part* part);
  void register_interior_algorithm(stk::mesh::Part* part);
  void execute();
  void initial_work();
  void initial_production();
  void initial_mdot();
  void compute_metric_tensor();

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

  FieldUpdateAlgDriver metricTensorAlgDriver_;
  std::unique_ptr<SSTTAMSAveragesAlg> avgAlg_;
  NgpAlgDriver avgMdotAlg_;

  const TurbulenceModel turbulenceModel_;

  bool resetTAMSAverages_;
};
} // namespace nalu
} // namespace sierra

#endif
