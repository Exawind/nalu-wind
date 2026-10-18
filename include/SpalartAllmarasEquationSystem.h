// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SpalartAllmarasEquationSystem_h
#define SpalartAllmarasEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <KynemaUGFParsedTypes.h>

#include "ngp_algorithms/NodalGradAlgDriver.h"

#include <memory>

namespace stk {
struct topology;
namespace mesh {
class Part;
}
} // namespace stk

namespace sierra {
namespace kynema_ugf {

class EquationSystems;
class AlgorithmDriver;
class ProjectedNodalGradientEquationSystem;

class SpalartAllmarasEquationSystem : public EquationSystem
{

public:
  SpalartAllmarasEquationSystem(EquationSystems& equationSystems);
  virtual ~SpalartAllmarasEquationSystem();

  virtual void load(const YAML::Node&);

  virtual void initialize();

  virtual void reinitialize_linear_system();

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);

  virtual void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const InflowBoundaryConditionData& inflowBCData);

  virtual void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const OpenBoundaryConditionData& openBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const SymmetryBoundaryConditionData& symmetryBCData);

  virtual void register_abltop_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const ABLTopBoundaryConditionData& ablTopBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_overset_bc();

  virtual void register_interior_algorithm(stk::mesh::Part* part);

  virtual void register_initial_condition_fcn(
    stk::mesh::Part* part,
    const std::map<std::string, std::string>& theNames,
    const std::map<std::string, std::vector<double>>& theParams);

  virtual void solve_and_update();

  virtual void predict_state();

  void compute_effective_diff_flux_coeff();
  void compute_wall_model_parameters();
  void compute_projected_nodal_gradient();
  void update_nu_tilda();
  void clip_min_distance_to_wall();
  void manage_projected_nodal_gradient(EquationSystems& eqSystems);

  bool managePNG_;
  bool isInit_;

  ScalarFieldType* nuTilda_;
  VectorFieldType* dnutdx_;
  ScalarFieldType* nuTmp_;
  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* evisc_;
  ScalarFieldType* minDistanceToWall_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  std::unique_ptr<Algorithm> effDiffFluxAlg_;
  ProjectedNodalGradientEquationSystem* projectedNodalGradEqs_;

  // saved of mesh parts that are for wall bcs
  std::vector<stk::mesh::Part*> wallBcPart_;

  bool wallFunctionWarningIssued_{false};
};

} // namespace kynema_ugf
} // namespace sierra

#endif
