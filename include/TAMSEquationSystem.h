/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TAMSEquationSystem_h
#define TAMSEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
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

class AlgorithmDriver;
class Realm;
class AssembleNodalGradAlgorithmDriver;
class LinearSystem;
class EquationSystems;

class TAMSEquationSystem : public EquationSystem
{

public:
  using DblType = double;

  TAMSEquationSystem(EquationSystems& equationSystems);
  virtual ~TAMSEquationSystem() = default;

  virtual void register_nodal_fields(stk::mesh::Part* part);

  virtual void
  register_element_fields(stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_edge_fields(stk::mesh::Part* part);

  void register_interior_algorithm(stk::mesh::Part* part);

  void initial_work();
  void pre_iter_work();
  void compute_metric_tensor();
  void compute_averages();
  void compute_avgMdot();
  // void load(const YAML::Node & node);
  void predict_state();

  const bool managePNG_;

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
  std::unique_ptr<SSTTAMSAveragesAlg> avgAlg_{nullptr};
  std::unique_ptr<Algorithm> tviscAlg_{nullptr};
  NgpAlgDriver avgMdotAlg_;

  const TurbulenceModel turbulenceModel_;

  bool resetTAMSAverages_;
};

} // namespace nalu
} // namespace sierra

#endif
