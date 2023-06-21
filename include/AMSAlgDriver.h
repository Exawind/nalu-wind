// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AMSAlgDriver_h
#define AMSAlgDriver_h

#include "FieldTypeDef.h"
#include "Algorithm.h"
#include "ngp_algorithms/NgpAlgDriver.h"
#include "ngp_algorithms/FieldUpdateAlgDriver.h"
#include "ngp_algorithms/AMSAvgMdotEdgeAlg.h"
#include "ngp_algorithms/SSTAMSAveragesAlg.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class Realm;

class AMSAlgDriver
{

public:
  using DblType = double;

  AMSAlgDriver(Realm& realm);
  virtual ~AMSAlgDriver() = default;
  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);
  virtual void register_edge_fields(const stk::mesh::PartVector& part_vec);
  void register_interior_algorithm(stk::mesh::Part* part);
  void execute();
  void initial_work();
  void initial_production();
  void initial_mdot();
  void compute_metric_tensor();
  void predict_state();
  void post_iter_work();

private:
  Realm& realm_;

  VectorFieldType* avgVelocity_;
  VectorFieldType* avgVelocityRTM_;
  ScalarFieldType* avgTkeResolved_;
  GenericFieldType* avgDudx_;
  GenericFieldType* metric_;
  ScalarFieldType* beta_;

  ScalarFieldType* resAdequacy_;
  ScalarFieldType* avgResAdequacy_;
  ScalarFieldType* avgProduction_;
  ScalarFieldType* avgTime_;
  ScalarFieldType* avgMdot_;
  VectorFieldType* forcingComp_;

  FieldUpdateAlgDriver metricTensorAlgDriver_;
  std::unique_ptr<SSTAMSAveragesAlg> avgAlg_;
  NgpAlgDriver avgMdotAlg_;

  const TurbulenceModel turbulenceModel_;

  bool resetAMSAverages_;
};
} // namespace nalu
} // namespace sierra

#endif
