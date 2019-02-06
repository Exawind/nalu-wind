/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TAMSEquationSystem_h
#define TAMSEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

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
  TAMSEquationSystem(EquationSystems& equationSystems);
  virtual ~TAMSEquationSystem();

  virtual void register_nodal_fields(stk::mesh::Part* part);

  virtual void
  register_element_fields(stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_edge_fields(stk::mesh::Part* part);

  void register_interior_algorithm(stk::mesh::Part* part);

  void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const InflowBoundaryConditionData& inflowBCData);

  void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const OpenBoundaryConditionData& openBCData);

  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const SymmetryBoundaryConditionData& symmetryBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();

  void solve_and_update();
  void initial_work();
  void post_converged_work();
  void initialize_mdot();
  void compute_metric_tensor();
  void compute_averages();
  void compute_avgMdot();
  void update_and_clip();

  const bool managePNG_;

  VectorFieldType* avgVelocity_;
  ScalarFieldType* avgDensity_;
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
  VectorFieldType* gTmp_;

  AlgorithmDriver* metricTensorAlgDriver_;
  AlgorithmDriver* averagingAlgDriver_;
  AlgorithmDriver* avgMdotAlgDriver_;
  AlgorithmDriver* tviscAlgDriver_;

  const TurbulenceModel turbulenceModel_;

  bool resetAverages_;
};

} // namespace nalu
} // namespace sierra

#endif
